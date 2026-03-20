# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from aster import ir, utils
from aster.layout import Layout
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.testing import (
    compile_mlir_module_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)

MCPU = "gfx942"

# MFMA 16x16x16 f16 fragment layouts (first-mode-slowest).
# A/B: (4 groups of 16 lanes) x 8 bytes per group, 32 bytes between groups.
MFMA_AB_LAYOUT = Layout(sizes=(4, 16), strides=(8, 32))
# C/D: (4 groups of 16 lanes) x 16 bytes per group, 64 bytes between groups.
MFMA_C_LAYOUT = Layout(sizes=(4, 16), strides=(16, 64))


def _build_mfma_kernel(name, target=MCPU, isa="cdna3"):
    """Single-wave MFMA 16x16x16 f16: D = A @ B^T + C."""
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C/D
    a_ptr, b_ptr, c_ptr = b.load_args()

    tid = b.global_thread_id()

    # Note: global load directly in mfma layout is not coalesced, this is a correctness check.
    ab_off = b.layout_byte_offset(tid, MFMA_AB_LAYOUT)
    c_off = b.layout_byte_offset(tid, MFMA_C_LAYOUT)

    a_addr = b.global_addr(a_ptr, ab_off)
    b_addr = b.global_addr(b_ptr, ab_off)
    c_addr = b.global_addr(c_ptr, c_off)

    a_frag = b.global_load_dwordx2(a_addr)
    b_frag = b.global_load_dwordx2(b_addr)
    b.wait_vmcnt(0)

    acc = b.init_agprx4(b.constant_i32(0))
    acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)

    b.global_store_dwordx4(acc, c_addr)
    b.wait_vmcnt(0)
    return b.build()


def _run_mfma_test(name):
    A = (np.random.default_rng(42).standard_normal(16 * 16) * 0.1).astype(np.float16)
    B = (np.random.default_rng(43).standard_normal(16 * 16) * 0.1).astype(np.float16)
    C = np.zeros(16 * 16, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_mfma_kernel(name)
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
            input_args=[A, B],
            output_args=[C],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(1, 1, 1),
            block_dim=(64, 1, 1),
        )

    # MFMA 16x16x16 computes D = A @ B^T (B stored row-major [N][K]).
    # Result is transposed because MFMA C layout stores m_group in lane//16.
    ref = (A.reshape(16, 16) @ B.reshape(16, 16).T).T
    np.testing.assert_allclose(C.reshape(16, 16), ref, rtol=1e-2, atol=1e-2)


def test_mfma_16x16x16_f16():
    _run_mfma_test("mfma_16x16")
