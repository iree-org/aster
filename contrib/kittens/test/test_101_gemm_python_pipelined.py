"""Pure-Python pipelined GEMM using layout-first programming model.

Supersedes test_005_gemm_fp16_lds_pipelined.mlir -- no .mlir template needed. Pipeline
scheduling via b.stage() context manager + sched.stage attributes.
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
from aster.pass_pipelines import make_default_pass_pipeline

from kittens.layouts import (
    TILE_16x64,
    LDS_SWIZZLE,
    MFMA_FRAG_IN_TILE,
    make_global_tile_layout,
    make_mfma_c_layout,
)

MCPU = "gfx942"

STAGE_LOAD = 0
STAGE_WRITE = 1
STAGE_READ = 2
STAGE_COMPUTE = 3


def _build_gemm_pipelined(k, stride_ab):
    """Build pipelined GEMM kernel in Python with layout + stage annotations."""
    k_tiles = k // 32
    stride_c = 16 * 4
    d0 = ir.AffineExpr.get_dim(0)
    GLOBAL_LAYOUT = make_global_tile_layout(stride_ab)
    C_LAYOUT = make_mfma_c_layout(stride_c)

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined", target=MCPU, isa="cdna3")
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()
    lane = b.lane_id()

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    acc_init = b.init_agprx4(b.constant_i32(0))

    def k_body(k_iv, acc):
        tile_off = b.affine_apply(d0 * 64, [k_iv])

        with b.stage(STAGE_LOAD):
            lds_a_h, lds_a = b.alloc_lds(1024)
            lds_b_h, lds_b = b.alloc_lds(1024)
            a_data, a_tok = b.tile_load(lane, a_ptr, tile_off, GLOBAL_LAYOUT)
            b_data, b_tok = b.tile_load(lane, b_ptr, tile_off, GLOBAL_LAYOUT)

        with b.stage(STAGE_WRITE):
            b.wait_deps(a_tok, b_tok)
            a_wtoks = b.tile_to_lds(lane, a_data, lds_a, TILE_16x64, LDS_SWIZZLE)
            b_wtoks = b.tile_to_lds(lane, b_data, lds_b, TILE_16x64, LDS_SWIZZLE)

        with b.stage(STAGE_READ):
            b.wait_deps(*a_wtoks, *b_wtoks)
            a0, a0t = b.frag_from_lds(lane, lds_a, MFMA_FRAG_IN_TILE, LDS_SWIZZLE)
            b0, b0t = b.frag_from_lds(lane, lds_b, MFMA_FRAG_IN_TILE, LDS_SWIZZLE)
            k1_off = b.constant_index(32)
            a1, a1t = b.frag_from_lds(
                lane, lds_a, MFMA_FRAG_IN_TILE, LDS_SWIZZLE, k1_off
            )
            b1, b1t = b.frag_from_lds(
                lane, lds_b, MFMA_FRAG_IN_TILE, LDS_SWIZZLE, k1_off
            )

        with b.stage(STAGE_COMPUTE):
            b.wait_deps(a0t, b0t, a1t, b1t)
            acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a0, b0)
            acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a1, b1)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)

        return [acc]

    [acc_final] = b.for_loop(c0, b.constant_index(k_tiles), c1, [acc_init], k_body)
    b.store_c_tile(lane, acc_final, c_ptr, b.constant_index(0), C_LAYOUT)
    return b.build()


class TestPythonGEMMPipelined:

    @pytest.mark.parametrize("k", [128, 256, 512, 1024])
    def test_gemm_pipelined_python(self, k):
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm_pipelined(k, stride_ab)
            asm = compile_mlir_module_to_asm(
                module, pass_pipeline=make_default_pass_pipeline()
            )

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
                kernel_name="gemm_pipelined",
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

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_pipelined(args.k, args.k * 2, 2)
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(),
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=args.print_ir_after_all,
                print_asm=args.print_asm,
            ),
        )
    if args.print_asm:
        print(asm)
