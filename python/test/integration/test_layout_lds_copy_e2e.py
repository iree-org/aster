# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from aster import ir, utils
from aster.layout import Layout, Swizzle
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.testing import (
    compile_mlir_module_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)

N_THREADS = 64
ELEM_BYTES = 8  # 2 f32 per thread (dwordx2 for ds_write_b64)
MCPU = "gfx942"


LDS_SIZE = N_THREADS * ELEM_BYTES
LINEAR = Layout(sizes=N_THREADS, strides=ELEM_BYTES)


def _build_lds_copy_kernel(name, layout, swizzle=None, target=MCPU, isa="cdna3"):
    """Global load -> LDS write (swizzled) -> LDS read -> global store."""
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.set_shared_memory_size(LDS_SIZE)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()
    tid = b.thread_id("x")

    src_voff = b.index_to_vgpr(b.linearize_layout(tid, layout))
    lds_voff = b.index_to_vgpr(b.linearize_layout(tid, layout, swizzle))

    dst_voff = b.index_to_vgpr(b.linearize_layout(tid, LINEAR))

    num_records = b.s_mov_b32(65536)
    soffset = b.s_mov_b32(0)
    src_rsrc = b.make_buffer_rsrc(src_ptr, num_records, b.constant_i32(0))
    dst_rsrc = b.make_buffer_rsrc(dst_ptr, num_records, b.constant_i32(0))

    data = b.buffer_load_dwordx2(src_rsrc, soffset, src_voff)
    b.wait_vmcnt(0)
    b.ds_write_b64(data, lds_voff)
    b.wait_lgkmcnt(0)
    data_from_lds = b.ds_read_b64(lds_voff)
    b.wait_lgkmcnt(0)
    b.buffer_store_dwordx2(data_from_lds, dst_rsrc, soffset, dst_voff)
    b.wait_vmcnt(0)
    return b.build()


def _run_lds_copy_test(name, layout, swizzle=None):
    elems_per = ELEM_BYTES // 4
    offsets = [swizzle(layout(i)) if swizzle else layout(i) for i in range(N_THREADS)]

    src = np.zeros((max(offsets) + ELEM_BYTES) // 4, dtype=np.float32)
    for tid, off in enumerate(offsets):
        for j in range(elems_per):
            src[off // 4 + j] = tid * 10.0 + j

    dst = np.zeros(N_THREADS * elems_per, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_lds_copy_kernel(name, layout, swizzle)
        asm = compile_mlir_module_to_asm(module)

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
            block_dim=(N_THREADS, 1, 1),
        )

    for tid in range(N_THREADS):
        for j in range(elems_per):
            actual = dst[tid * elems_per + j]
            expected = tid * 10.0 + j
            assert (
                actual == expected
            ), f"tid={tid} elem={j}: got {actual}, expected {expected}"


def test_lds_copy_linear():
    """LDS round-trip with linear layout, no swizzle."""
    _run_lds_copy_test("lds_lin", LINEAR)


def test_lds_copy_swizzled():
    """LDS round-trip with swizzled addresses for bank conflict avoidance."""
    # base=4 > log2(ELEM_BYTES=8)=3 so no intra-element aliasing.
    _run_lds_copy_test("lds_swz", LINEAR, Swizzle(bits=2, base=4, shift=6))
