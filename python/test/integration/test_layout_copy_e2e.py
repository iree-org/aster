# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from aster import ir
from aster.layout import Layout, Swizzle
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu

ELEM_BYTES = 16  # 4 f32 per thread (dwordx4)
MCPU = "gfx942"


def _build_copy_kernel(
    name,
    src_layout,
    n_threads,
    swizzle=None,
    use_global=False,
    target=MCPU,
):
    """Copy kernel: load dwordx4 at layout(+swizzle) offset, store linear.

    If use_global=True, uses global load/store (ptr + offset -> VGPRx2).
    Otherwise uses buffer load/store (rsrc + voffset).
    """
    b = KernelBuilder(f"{name}_mod", name, target=target)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    tid = b.global_thread_id()

    src_off = b.linearize_layout(tid, src_layout, swizzle)
    linear = Layout(sizes=n_threads, strides=ELEM_BYTES)
    dst_off = b.linearize_layout(tid, linear)

    if use_global:
        src_addr = b.global_addr(src_ptr, src_off)
        dst_addr = b.global_addr(dst_ptr, dst_off)
        data, load_tok = b.global_load_dwordx4(src_addr)
        b.wait_deps(load_tok)
        b.global_store_dwordx4(data, dst_addr)
    else:
        src_voff = b.index_to_vgpr(src_off)
        dst_voff = b.index_to_vgpr(dst_off)
        num_records = b.s_mov_b32(n_threads * ELEM_BYTES)
        soffset = b.s_mov_b32(0)
        src_rsrc = b.make_buffer_rsrc(src_ptr, num_records, b.constant_i32(0))
        dst_rsrc = b.make_buffer_rsrc(dst_ptr, num_records, b.constant_i32(0))
        data = b.buffer_load_dwordx4(src_rsrc, soffset, src_voff)
        b.wait_vmcnt(0)
        b.buffer_store_dwordx4(data, dst_rsrc, soffset, dst_voff)

    b.wait_vmcnt(0)
    return b.build()


def _run_copy_test(
    name,
    layout,
    n_threads,
    swizzle=None,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
    use_global=False,
):
    elems_per = ELEM_BYTES // 4
    offsets = [swizzle(layout(i)) if swizzle else layout(i) for i in range(n_threads)]

    src = np.zeros((max(offsets) + ELEM_BYTES) // 4, dtype=np.float32)
    for tid, off in enumerate(offsets):
        for j in range(elems_per):
            src[off // 4 + j] = tid * 10.0 + j

    dst = np.zeros(n_threads * elems_per, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_copy_kernel(name, layout, n_threads, swizzle, use_global)
        asm = compile_mlir_module_to_asm(module)

    path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler not compiled with {MCPU} support (unknown target)")

    with hsaco_file(path):
        if not system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=name,
            arguments=[InputArray(src), OutputArray(dst)],
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

    for tid in range(n_threads):
        for j in range(elems_per):
            actual = dst[tid * elems_per + j]
            expected = tid * 10.0 + j
            assert actual == expected, (
                f"tid={tid} elem={j}: got {actual}, expected {expected}"
            )


@pytest.mark.parametrize("use_global", [False, True], ids=["buffer", "flat"])
@pytest.mark.parametrize(
    "name, layout, swizzle",
    [
        ("linear", Layout(sizes=64, strides=ELEM_BYTES), None),
        (
            "tiled",
            Layout(sizes=(4, 16), strides=(16 * ELEM_BYTES, ELEM_BYTES)),
            None,
        ),
        (
            "swizzled",
            Layout(sizes=64, strides=ELEM_BYTES),
            Swizzle(bits=2, base=4, shift=6),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_copy_single_wg(name, layout, swizzle, use_global):
    _run_copy_test(
        f"cp_{name}",
        layout,
        n_threads=64,
        swizzle=swizzle,
        use_global=use_global,
    )


# ---------------------------------------------------------------------------
# Multi-WG nested layout stress tests
# ---------------------------------------------------------------------------
# Grid: n_wg_x=8, n_wg_y=16 (128 WGs). Block: n_wv_x=2 * n_wv_y=4 waves x 64 lanes = 512.
# Total: 8 * 16 * 2 * 4 * 64 = 65536 threads.
# Strides are contiguous (coalesceable) so all nestings produce the same
# function -- the test verifies the compiler handles each nesting correctly.

# Powers of 2 here; non-power-of-2 divisors also supported via SDivByConstant.
N_WV_X, N_WV_Y = 2, 4
N_WG_X, N_WG_Y = 8, 16
LANE = 64
TPW = N_WV_X * N_WV_Y * LANE  # 2*4*64 = 512
TOTAL = N_WG_X * N_WG_Y * TPW  # 8*16*512 = 65536
# Contiguous strides (first-mode-slowest: wg_y slowest, lane fastest).
LANE_S = ELEM_BYTES  # 16
WV_X_S = LANE * LANE_S  # 1024
WV_Y_S = N_WV_X * WV_X_S  # 2048
WG_X_S = N_WV_Y * WV_Y_S  # 4096
WG_Y_S = N_WG_X * WG_X_S  # 8192


@pytest.mark.parametrize("use_global", [False, True], ids=["buffer", "flat"])
@pytest.mark.parametrize(
    "name, layout",
    [
        # Flat rank-1
        ("flat", Layout(sizes=TOTAL, strides=LANE_S)),
        # Flat rank-5: (wg_y, wg_x, wv_y, wv_x, lane)
        (
            "flat5",
            Layout(
                sizes=(N_WG_Y, N_WG_X, N_WV_Y, N_WV_X, LANE),
                strides=(WG_Y_S, WG_X_S, WV_Y_S, WV_X_S, LANE_S),
            ),
        ),
        # Nested: ((wg_y, wg_x), ((wv_y, wv_x), lane))
        (
            "wg_wv_lane",
            Layout(
                sizes=((N_WG_Y, N_WG_X), ((N_WV_Y, N_WV_X), LANE)),
                strides=((WG_Y_S, WG_X_S), ((WV_Y_S, WV_X_S), LANE_S)),
            ),
        ),
        # Deeply nested: (wg_y, (wg_x, (wv_y, (wv_x, lane))))
        (
            "deep",
            Layout(
                sizes=(N_WG_Y, (N_WG_X, (N_WV_Y, (N_WV_X, LANE)))),
                strides=(WG_Y_S, (WG_X_S, (WV_Y_S, (WV_X_S, LANE_S)))),
            ),
        ),
        # Two-level: ((wg_y, wg_x), (wv_y, wv_x, lane))
        (
            "wg_pair",
            Layout(
                sizes=((N_WG_Y, N_WG_X), (N_WV_Y, N_WV_X, LANE)),
                strides=((WG_Y_S, WG_X_S), (WV_Y_S, WV_X_S, LANE_S)),
            ),
        ),
        # Flat WG, nested wave: (wg_y, wg_x, ((wv_y, wv_x), lane))
        (
            "flat_wg_nest_wv",
            Layout(
                sizes=(N_WG_Y, N_WG_X, ((N_WV_Y, N_WV_X), LANE)),
                strides=(WG_Y_S, WG_X_S, ((WV_Y_S, WV_X_S), LANE_S)),
            ),
        ),
        # X/Y permutations: swap X before Y with matching strides.
        (
            "deep_xy_swap",
            Layout(
                sizes=(N_WG_X, (N_WG_Y, (N_WV_X, (N_WV_Y, LANE)))),
                strides=(WG_X_S, (WG_Y_S, (WV_X_S, (WV_Y_S, LANE_S)))),
            ),
        ),
        (
            "flat5_xy_swap",
            Layout(
                sizes=(N_WG_X, N_WG_Y, N_WV_X, N_WV_Y, LANE),
                strides=(WG_X_S, WG_Y_S, WV_X_S, WV_Y_S, LANE_S),
            ),
        ),
        (
            "pairs_xy_swap",
            Layout(
                sizes=((N_WG_X, N_WG_Y), ((N_WV_X, N_WV_Y), LANE)),
                strides=((WG_X_S, WG_Y_S), ((WV_X_S, WV_Y_S), LANE_S)),
            ),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_copy_multiwg_nested(name, layout, use_global):
    _run_copy_test(
        f"cp_{name}",
        layout,
        n_threads=TOTAL,
        grid_dim=(N_WG_X, N_WG_Y, 1),
        block_dim=(TPW, 1, 1),
        use_global=use_global,
    )
