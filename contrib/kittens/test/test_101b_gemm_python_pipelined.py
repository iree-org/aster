"""Pipelined GEMM with num_tiles_per_wg.

Hill-climb from test_101: one WG, one wave owns the kernel.
The wave produces an (m_t * 16) x (n_t * 16) C tile and walks K in k_t * 32
chunks per outer K-iter.
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
    global_store_dword,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from aster.layout import Layout, LayoutValues, Swizzle, Symbol, Tensor, enumerate_flat_coords, tile


# Layout symbols: ``None`` marks fixed modes; only named symbols are bindable.
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")

MCPU = "gfx942"

# Per-tile fixed geometry.
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
TILE_K_ELEMS = 32  # K elements per LDS-tile transfer
ELT_BYTES = 2  # fp16
TILE_BYTES_A = MFMA_M * TILE_K_ELEMS * ELT_BYTES  # 1024
TILE_BYTES_B = MFMA_N * TILE_K_ELEMS * ELT_BYTES  # 1024

# Pipeline stages (mirrors strategy 6 in kittens_helpers PIPELINE_STRATEGIES).
STG_A_LOAD = 0
STG_B_LOAD = 1
STG_B_WRITE = 1
STG_A_WRITE = 2
STG_A_READ = 2
STG_B_READ = 2
STG_COMPUTE = 3
N_STAGES = STG_COMPUTE + 1


def _build_gemm_pipelined(num_tiles_per_wg, K, stride_a, stride_b):
    """Build a single-wave pipelined GEMM with num_tiles_per_wg tiles."""
    m_t, n_t, k_t = num_tiles_per_wg

    lds_total_a = k_t * m_t * TILE_BYTES_A
    lds_total_b = k_t * n_t * TILE_BYTES_B

    k_step = k_t * TILE_K_ELEMS
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, (
        f"k_iters={k_iters} < N_STAGES={N_STAGES}; pipeliner needs at least N_STAGES iters for prologue+steady-state"
    )

    M, N = m_t * MFMA_M, n_t * MFMA_N
    stride_c = N * 4

    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined", target=MCPU)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((MFMA_M, 4), (stride_a, 16)),
        value_layout=Layout(1, 0),
    )
    tc_load_b = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((MFMA_N, 4), (stride_b, 16)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((MFMA_M, 4), (64, 16)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, 8),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsw_b = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((MFMA_N, 4), (64, 16)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, 8),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, MFMA_M), (8, 64)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, MFMA_K * ELT_BYTES),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_b = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, MFMA_N), (8, 64)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, MFMA_K * ELT_BYTES),
        swizzle=LDS_SWIZZLE,
    )
    # C store: MFMA 16x16 accumulator -> row-major fp32 C tile. The lane owns
    # col = lane%16 and the 4-row block starting at (lane//16)*4; value v is
    # the row within that block (4 fp32 D-registers per lane).
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((4, MFMA_N), (4 * stride_c, 4)),
        value_layout=Layout(4, stride_c),
    )

    # Global A/B tiled. Outer K becomes the per-iter axis (global_k); inner
    # is the per-iter K-tile axis (None = unnamed, iterated by transfer_tiles).
    A_TILED = tile(
        Layout((M, K), (stride_a, ELT_BYTES)),
        tile_sizes=(MFMA_M, (TILE_K_ELEMS, k_t)),
        axes=(m, (k_tile, global_k)),
    )
    B_TILED = tile(
        Layout((N, K), (stride_b, ELT_BYTES)),
        tile_sizes=(MFMA_N, (TILE_K_ELEMS, k_t)),
        axes=(n, (k_tile, global_k)),
    )
    C_TILED = tile(
        Layout((M, N), (stride_c, 4)),
        tile_sizes=(MFMA_M, MFMA_N),
        axes=(m, n),
    )

    # LDS A/B: each tile is one TILE_BYTES_{A,B} byte chunk; m-outer, k-inner.
    LDS_A_TILED = tile(
        Layout((m_t, k_t * TILE_BYTES_A), (k_t * TILE_BYTES_A, 1)),
        tile_sizes=(1, TILE_BYTES_A),
        axes=(m, k_tile),
    )
    LDS_B_TILED = tile(
        Layout((n_t, k_t * TILE_BYTES_B), (k_t * TILE_BYTES_B, 1)),
        tile_sizes=(1, TILE_BYTES_B),
        axes=(n, k_tile),
    )

    TA = Tensor(a_ptr, layout=A_TILED)
    TB = Tensor(b_ptr, layout=B_TILED)
    TC = Tensor(c_ptr, layout=C_TILED)

    n_accs = m_t * n_t
    n_frags = TILE_K_ELEMS // MFMA_K  # MFMA fragments per ds_read tile
    acc_inits = [b.init_agprx4(b.constant_i32(0)) for _ in range(n_accs)]
    accs_final = []

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)

    @b.loop(c0, b.constant_index(k_iters), c1, iter_args=acc_inits, results=accs_final)
    def body(k_iv, *accs):
        accs = list(accs)
        with b.stage(STG_A_LOAD):
            lds_a_h, sA = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_TILED)
            a_load = b.transfer_tiles(b.slice(TA, {global_k: k_iv}), tc_load_a, unroll_axes=(m, k_tile))

        with b.stage(STG_B_LOAD):
            lds_b_h, sB = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_TILED)
            b_load = b.transfer_tiles(b.slice(TB, {global_k: k_iv}), tc_load_b, unroll_axes=(n, k_tile))

        with b.stage(STG_A_WRITE):
            b.wait_deps(a_load)
            a_write = b.transfer_tiles(sA, tc_dsw_a, unroll_axes=(m, k_tile), data=a_load)

        with b.stage(STG_B_WRITE):
            b.wait_deps(b_load)
            b_write = b.transfer_tiles(sB, tc_dsw_b, unroll_axes=(n, k_tile), data=b_load)

        with b.stage(STG_A_READ):
            b.wait_deps(a_write)
            a_frags = b.transfer_tiles(sA, tc_dsr_a, unroll_axes=(m, k_tile))

        with b.stage(STG_B_READ):
            b.wait_deps(b_write)
            b_frags = b.transfer_tiles(sB, tc_dsr_b, unroll_axes=(n, k_tile))

        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_frags)
            # Note: in IR this would be scf.for + aster.constexpr + delinearize
            # Order matches b.delinearize_index((nf, k_t, m_t, n_t))
            for fi, ki, mi, ni in enumerate_flat_coords((n_frags, k_t, m_t, n_t)):
                ai = mi * n_t + ni
                a_d = a_frags.data_at((mi, ki, fi))
                b_d = b_frags.data_at((ni, ki, fi))
                accs[ai] = b.mfma("v_mfma_f32_16x16x16_f16", accs[ai], a_d, b_d)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)
        return accs

    # C store: one tiled copy per (m, n) tile, mirroring the load path.
    accs_bundle = LayoutValues.from_flat(Layout((m_t, n_t)), payloads=tuple(accs_final))
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(m, n), data=accs_bundle)
    return b.build()


class TestPythonGEMMPipelined:
    @pytest.mark.parametrize(
        "num_tiles_per_wg",
        [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (3, 2, 1),
            (1, 1, 2),
            (2, 2, 2),
        ],
        ids=["1x1x1", "2x1x1", "1x2x1", "2x2x1", "3x2x1", "1x1x2", "2x2x2"],
    )
    @pytest.mark.parametrize("K", [256, 512])
    def test_gemm_pipelined_python(self, num_tiles_per_wg, K):
        m_t, n_t, k_t = num_tiles_per_wg
        M, N = m_t * MFMA_M, n_t * MFMA_N
        stride_a = K * ELT_BYTES
        stride_b = K * ELT_BYTES

        np.random.seed(42 + K + M + N)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N, K) * 0.1).astype(np.float16)
        C_output = np.zeros(M * N, dtype=np.float32)

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm_pipelined(num_tiles_per_wg, K, stride_a, stride_b)
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
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--m_t", type=int, default=1)
    parser.add_argument("--n_t", type=int, default=1)
    parser.add_argument("--k_t", type=int, default=1)
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    num_tiles_per_wg = (args.m_t, args.n_t, args.k_t)
    stride_a = args.K * ELT_BYTES
    stride_b = args.K * ELT_BYTES
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_pipelined(num_tiles_per_wg, args.K, stride_a, stride_b)
        if args.print_ir_after_all:
            print(module)
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
