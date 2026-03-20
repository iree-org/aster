#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from aster import ir, utils
from aster.layout import Layout, Swizzle
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects import amdgcn as amdgcn_dialect
from aster.dialects.amdgcn import AccessKind
from aster.testing import execute_kernel_and_verify, hsaco_file

ELEM_BYTES = 16  # 4 f32 per thread (dwordx4)
MCPU = "gfx942"


def _build_copy_kernel(
    name,
    src_layout,
    n_threads,
    swizzle=None,
    target=MCPU,
    isa="cdna3",
):
    """Copy kernel: load dwordx4 at layout(+swizzle) offset, store linear."""
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    tid = b.thread_id_x()

    src_voff = b.index_to_vgpr(b.layout_byte_offset(tid, src_layout, swizzle))

    linear = Layout(sizes=n_threads, strides=ELEM_BYTES)
    dst_voff = b.index_to_vgpr(b.layout_byte_offset(tid, linear))

    num_records = b.s_mov_b32(65536)
    soffset = b.s_mov_b32(0)
    src_rsrc = b.make_buffer_rsrc(src_ptr, num_records, b.constant_i32(0))
    dst_rsrc = b.make_buffer_rsrc(dst_ptr, num_records, b.constant_i32(0))

    data = b.buffer_load_dwordx4(src_rsrc, soffset, src_voff)
    b.wait_vmcnt(0)
    b.buffer_store_dwordx4(data, dst_rsrc, soffset, dst_voff)
    b.wait_vmcnt(0)
    return b.build()


def _compile_to_asm(module):
    from aster._mlir_libs._mlir import passmanager
    from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

    ctx = ir.Context.current
    pm = passmanager.PassManager.parse(TEST_SROA_PASS_PIPELINE, ctx)
    pm.run(module.operation)
    for op in module.body:
        if isinstance(op, amdgcn_dialect.ModuleOp):
            return utils.translate_module(op)
    raise RuntimeError("No amdgcn.module found")


def _run_copy_test(name, layout, n_threads, swizzle=None):
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
        module = _build_copy_kernel(name, layout, n_threads, swizzle)
        asm = _compile_to_asm(module)

    path = utils.assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {MCPU}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")
        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name=name,
            input_args=[src],
            output_args=[dst],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(1, 1, 1),
            block_dim=(n_threads, 1, 1),
        )

    for tid in range(n_threads):
        for j in range(elems_per):
            actual = dst[tid * elems_per + j]
            expected = tid * 10.0 + j
            assert (
                actual == expected
            ), f"tid={tid} elem={j}: got {actual}, expected {expected}"


def test_copy_linear():
    """Linear layout: tid * 16 bytes."""
    _run_copy_test("copy_lin", Layout(sizes=64, strides=ELEM_BYTES), n_threads=64)


def test_copy_tiled():
    """Tiled layout: 4 tiles of 16 threads, first-mode-slowest."""
    tiled = Layout(sizes=(4, 16), strides=(16 * ELEM_BYTES, ELEM_BYTES))
    _run_copy_test("copy_tile", tiled, n_threads=64)


def test_copy_swizzled():
    """Linear + swizzle on source offset."""
    _run_copy_test(
        "copy_swz",
        Layout(sizes=64, strides=ELEM_BYTES),
        n_threads=64,
        swizzle=Swizzle(bits=2, base=4, shift=6),
    )
