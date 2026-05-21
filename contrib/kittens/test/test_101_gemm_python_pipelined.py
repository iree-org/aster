"""Pure-Python pipelined GEMM using layout-first programming model.

Supersedes test_005_gemm_fp16_lds_pipelined.mlir, no .mlir template
needed. Pipeline scheduling via b.stage() context manager + sched.stage
attributes. 16x16x(32 * k) problem size, 1 mxn tile per wave.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder_with_layouts import (
    ds_read_64b,
    ds_write_64b,
    global_load_dwordx4,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from aster.layout import Layout, Swizzle, Tensor

LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

MCPU = "gfx942"

STAGE_LOAD = 0
STAGE_WRITE = 1
STAGE_READ = 2
STAGE_COMPUTE = 3


def _build_gemm_pipelined(k):
    """Build pipelined GEMM kernel in Python with layout + stage annotations."""
    k_tiles = k // 32
    stride_a = k * 2
    stride_b = k * 2
    stride_c = 16 * 4
    d0 = ir.AffineExpr.get_dim(0)
    GLOBAL_STORE_TILE_C = Layout((4, 16, 4), (4 * stride_c, 4, stride_c))
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined", target=MCPU)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((16, 4), (stride_a, 16)),
        value_layout=Layout(1, 0),
    )
    tc_load_b = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((16, 4), (stride_b, 16)),
        value_layout=Layout(1, 0),
    )
    tc_dsw = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((16, 4), (64, 16)),
        value_layout=Layout(2, 8),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, 16), (8, 64)),
        value_layout=Layout(2, 32),
        swizzle=LDS_SWIZZLE,
    )

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    acc_init = b.init_agprx4(b.constant_i32(0))

    acc = []

    @b.loop(c0, b.constant_index(k_tiles), c1, iter_args=[acc_init], results=acc)
    def _(k_iv, acc):
        tile_off = b.affine_apply(d0 * 64, [k_iv])

        with b.stage(STAGE_LOAD):
            lds_a_h, lds_a = b.alloc_lds(1024)
            lds_b_h, lds_b = b.alloc_lds(1024)
            A_tile = Tensor(a_ptr, offset=tile_off)
            B_tile = Tensor(b_ptr, offset=tile_off)
            a_load_res = b.transfer_tile(A_tile, tc_load_a)
            b_load_res = b.transfer_tile(B_tile, tc_load_b)

        with b.stage(STAGE_WRITE):
            b.wait_deps(a_load_res, b_load_res)
            sA = Tensor(lds_a)
            sB = Tensor(lds_b)
            dsw_a_res = b.transfer_tile(sA, tc_dsw, data=b.split_vx4(a_load_res.data_at(0)))
            dsw_b_res = b.transfer_tile(sB, tc_dsw, data=b.split_vx4(b_load_res.data_at(0)))

        with b.stage(STAGE_READ):
            b.wait_deps(dsw_a_res, dsw_b_res)
            a_frags = b.transfer_tile(sA, tc_dsr)
            b_frags = b.transfer_tile(sB, tc_dsr)

        with b.stage(STAGE_COMPUTE):
            for (a_d, a_t), (b_d, b_t) in zip(a_frags, b_frags):
                b.wait_deps(a_t, b_t)
                acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_d, b_d)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)

        return [acc]

    [acc_final] = acc
    b.store_multi_fragment_to_global(
        acc_final,
        c_ptr,
        b.constant_index(0),
        GLOBAL_STORE_TILE_C,
        GLOBAL_STORE_SUB_TILE_C,
        b.global_store_dword,
    )
    return b.build()


class TestPythonGEMMPipelined:
    @pytest.mark.parametrize("k", [128, 256, 512, 1024])
    def test_gemm_pipelined_python(self, k):
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm_pipelined(k)
            asm = compile_mlir_module_to_asm(module, pass_pipeline=make_default_pass_pipeline(PipelineConfig()))

        path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
        if path is None:
            pytest.skip(f"LLVM assembler not compiled with {MCPU} support (unknown target)")

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
        module = _build_gemm_pipelined(args.k)
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=args.print_ir_after_all,
                print_asm=args.print_asm,
            ),
        )
    if args.print_asm:
        print(asm)
