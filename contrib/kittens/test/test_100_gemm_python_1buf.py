"""Pure-Python 1-buffer GEMM using layout-first programming model.

Supersedes test_001_gemm_fp16_lds_1buf.mlir -- no .mlir template needed. All address
computation expressed via Layout + Swizzle declarations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu

from kittens.layouts import (
    TILE_16x64,
    LDS_SWIZZLE,
    MFMA_FRAG_IN_TILE,
    make_global_tile_layout,
    make_mfma_c_layout,
)

MCPU = "gfx942"


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------


def _build_gemm_1buf(k, stride_ab):
    """Build 1-buffer GEMM kernel entirely in Python using layouts."""
    k_tiles = k // 32
    stride_c = 16 * 4  # 16 f32 columns * 4 bytes
    d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
    GLOBAL_LAYOUT = make_global_tile_layout(stride_ab)
    C_LAYOUT = make_mfma_c_layout(stride_c)

    b = KernelBuilder("gemm_1buf_mod", "gemm_1buf", target=MCPU, isa="cdna3")
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C
    a_ptr, b_ptr, c_ptr = b.load_args()
    lane = b.lane_id()

    lds_a_h, lds_a = b.alloc_lds(1024)
    lds_b_h, lds_b = b.alloc_lds(1024)

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    c_k_tiles = b.constant_index(k_tiles)
    acc_init = b.init_agprx4(b.constant_i32(0))

    def k_body(k_iv, acc):
        tile_off = b.affine_apply(d0 * 64, [k_iv])

        # Global load -> LDS write -> LDS read -> MFMA, all via layouts
        a_data, a_tok = b.tile_load(lane, a_ptr, tile_off, GLOBAL_LAYOUT)
        b_data, b_tok = b.tile_load(lane, b_ptr, tile_off, GLOBAL_LAYOUT)
        b.wait_deps(a_tok, b_tok)

        a_wtoks = b.tile_to_lds(lane, a_data, lds_a, TILE_16x64, LDS_SWIZZLE)
        b_wtoks = b.tile_to_lds(lane, b_data, lds_b, TILE_16x64, LDS_SWIZZLE)
        b.wait_deps(*a_wtoks, *b_wtoks)

        # K0 sub-tile (byte offset 0)
        a0, a0t = b.frag_from_lds(lane, lds_a, MFMA_FRAG_IN_TILE, LDS_SWIZZLE)
        b0, b0t = b.frag_from_lds(lane, lds_b, MFMA_FRAG_IN_TILE, LDS_SWIZZLE)
        b.wait_deps(a0t, b0t)
        acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a0, b0)

        # K1 sub-tile (byte offset 32)
        k1_off = b.constant_index(32)
        a1, a1t = b.frag_from_lds(lane, lds_a, MFMA_FRAG_IN_TILE, LDS_SWIZZLE, k1_off)
        b1, b1t = b.frag_from_lds(lane, lds_b, MFMA_FRAG_IN_TILE, LDS_SWIZZLE, k1_off)
        b.wait_deps(a1t, b1t)
        acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a1, b1)
        return [acc]

    [acc_final] = b.for_loop(c0, c_k_tiles, c1, [acc_init], k_body)

    b.store_c_tile(lane, acc_final, c_ptr, b.constant_index(0), C_LAYOUT)
    b.dealloc_lds(lds_a_h)
    b.dealloc_lds(lds_b_h)
    return b.build()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestPythonGEMM1Buffer:
    """Pure-Python 1-buffer GEMM using layouts -- no .mlir template."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_1buf_python(self, k):
        stride_ab = k * 2  # bytes per row (K f16 elements)

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm_1buf(k, stride_ab)
            asm = compile_mlir_module_to_asm(module)

        path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
        if path is None:
            pytest.skip(
                f"LLVM assembler not compiled with {MCPU} support (unknown target)"
            )

        with hsaco_file(path):
            if not system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"{MCPU} GPU not available")
            execute_hsaco(
                hsaco_path=path,
                kernel_name="gemm_1buf",
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B.flatten()),
                    OutputArray(C_output),
                ],
                grid_dim=(1, 1, 1),
                block_dim=(64, 1, 1),
            )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    k = args.k
    stride_ab = k * 2
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_1buf(k, stride_ab)
        asm = compile_mlir_module_to_asm(
            module,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=args.print_ir_after_all,
                print_asm=args.print_asm,
            ),
        )
    if args.print_asm:
        print(asm)
