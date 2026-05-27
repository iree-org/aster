"""Pipelined GEMM with multi-wave compute split, private per-wave LDS.

Hill-climb from test_101c: multi-WG, multi-wave per WG.
The waves produces an (m_t * 16) x (n_t * 16) C tile and walk K in k_t * 32
chunks per outer K-iter.
A 2D grid of (wg_m_count, wg_n_count) workgroups covers a larger M x N problem.
One C-tile per WG.

The per-WG (m_t x n_t x k_t) tile is split across nw = wpw_m * wpw_n waves
for the compute part.
Constraints: wpw_m must divide m_t, wpw_n must divide n_t.

Per-WG LDS layout: linear-wave-id partition into nw slots (one per wave).
Each wave's load/write/read all live entirely in its slot, so no
cross-wave LDS coherence is needed and there is no s_barrier in the loop.
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

from aster.layout import Layout, Swizzle, Symbol, Tensor, enumerate_flat_coords, tile


# Per-wave compute axes (same as test_101c).
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
# WG-level axes (bound from linear_block_id()).
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
# Per-WG wave-level axes for compute: which wave owns which sub-tile of
# the (m_t, n_t) WG output. wave_m partitions M; wave_n partitions N.
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# Linear wave id within the WG; used to partition LDS into nw private slots.
linear_wave_id = Symbol("linear_wave_id")

MCPU = "gfx942"
WAVE_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
TILE_K_ELEMS = 32
ELT_BYTES = 2
TILE_BYTES_A = MFMA_M * TILE_K_ELEMS * ELT_BYTES  # 1024
TILE_BYTES_B = MFMA_N * TILE_K_ELEMS * ELT_BYTES  # 1024

STG_A_LOAD = 0
STG_B_LOAD = 1
STG_B_WRITE = 1
STG_A_WRITE = 2
STG_A_READ = 2
STG_B_READ = 2
STG_COMPUTE = 3
N_STAGES = STG_COMPUTE + 1


def _build_gemm_pipelined(num_workgroups, num_waves_per_wg, num_tiles_per_wg, K, stride_a, stride_b):
    """Build a multi-WG, multi-wave pipelined GEMM with private per-wave LDS."""
    wg_m_count, wg_n_count = num_workgroups
    wpw_m, wpw_n = num_waves_per_wg
    m_t, n_t, k_t = num_tiles_per_wg
    nw = wpw_m * wpw_n
    num_threads = nw * WAVE_SIZE

    assert m_t % wpw_m == 0, f"m_t={m_t} not divisible by wpw_m={wpw_m}"
    assert n_t % wpw_n == 0, f"n_t={n_t} not divisible by wpw_n={wpw_n}"
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    # Private per-wave LDS: nw slots, each holding one wave's strip.
    lds_total_a = nw * m_per_wave * k_t * TILE_BYTES_A
    lds_total_b = nw * n_per_wave * k_t * TILE_BYTES_B

    k_step = k_t * TILE_K_ELEMS
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, (
        f"k_iters={k_iters} < N_STAGES={N_STAGES}; pipeliner needs at least N_STAGES iters for prologue+steady-state"
    )

    M_total = wg_m_count * m_t * MFMA_M
    N_total = wg_n_count * n_t * MFMA_N
    stride_c = N_total * 4

    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined_multiwave", target=MCPU)
    b.set_grid_dims(wg_m_count * wg_n_count)
    b.set_block_dims(num_threads)
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

    # Global A/B/C: hierarchical (MFMA, m_per_wave, wpw) splits the per-WG
    # tile across waves on M (or N). After slicing wg_* + wave_*, each wave
    # sees its own (m_per_wave, k_t) A strip and (n_per_wave, k_t) B strip.
    A_TILED = tile(
        Layout((M_total, K), (stride_a, ELT_BYTES)),
        tile_sizes=((MFMA_M, m_per_wave, wpw_m), (TILE_K_ELEMS, k_t)),
        axes=((m, wave_m, wg_m), (k_tile, global_k)),
    )
    B_TILED = tile(
        Layout((N_total, K), (stride_b, ELT_BYTES)),
        tile_sizes=((MFMA_N, n_per_wave, wpw_n), (TILE_K_ELEMS, k_t)),
        axes=((n, wave_n, wg_n), (k_tile, global_k)),
    )
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c, 4)),
        tile_sizes=((MFMA_M, m_per_wave, wpw_m), (MFMA_N, n_per_wave, wpw_n)),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

    # LDS layouts: nw private slots indexed by linear wave_id. Each slot is
    # a per-wave (m_per_wave x k_t * TILE_BYTES_A) (or *_B) strip.
    LDS_A_TILED = tile(
        Layout(
            (nw * m_per_wave, k_t * TILE_BYTES_A),
            (k_t * TILE_BYTES_A, 1),
        ),
        tile_sizes=((1, m_per_wave), TILE_BYTES_A),
        axes=((m, linear_wave_id), k_tile),
    )
    LDS_B_TILED = tile(
        Layout(
            (nw * n_per_wave, k_t * TILE_BYTES_B),
            (k_t * TILE_BYTES_B, 1),
        ),
        tile_sizes=((1, n_per_wave), TILE_BYTES_B),
        axes=((n, linear_wave_id), k_tile),
    )

    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg_m_count, wg_n_count))
    wid = b.wave_id(wave_size=WAVE_SIZE)
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    TA = b.slice(
        Tensor(a_ptr, layout=A_TILED),
        {wg_m: wg_m_idx, wave_m: wave_m_idx},
    )
    TB = b.slice(
        Tensor(b_ptr, layout=B_TILED),
        {wg_n: wg_n_idx, wave_n: wave_n_idx},
    )
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {
            wg_m: wg_m_idx,
            wg_n: wg_n_idx,
            wave_m: wave_m_idx,
            wave_n: wave_n_idx,
        },
    )

    GLOBAL_STORE_TILE_C = Layout((4, MFMA_N, 4), (4 * stride_c, 4, stride_c))
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    n_accs = m_per_wave * n_per_wave
    n_frags = TILE_K_ELEMS // MFMA_K
    acc_inits = [b.init_agprx4(b.constant_i32(0)) for _ in range(n_accs)]
    accs_final = []

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)

    @b.loop(c0, b.constant_index(k_iters), c1, iter_args=acc_inits, results=accs_final)
    def body(k_iv, *accs):
        accs = list(accs)
        with b.stage(STG_A_LOAD):
            lds_a_h, sA_full = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_TILED)
            sA = b.slice(sA_full, {linear_wave_id: wid})
            a_load = b.transfer_tiles(b.slice(TA, {global_k: k_iv}), tc_load_a, unroll_axes=(m, k_tile))

        with b.stage(STG_B_LOAD):
            lds_b_h, sB_full = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_TILED)
            sB = b.slice(sB_full, {linear_wave_id: wid})
            b_load = b.transfer_tiles(b.slice(TB, {global_k: k_iv}), tc_load_b, unroll_axes=(n, k_tile))

        with b.stage(STG_A_WRITE):
            b.wait_deps(a_load)
            a_write = b.transfer_tiles(sA, tc_dsw_a, unroll_axes=(m, k_tile), data=a_load)

        with b.stage(STG_B_WRITE):
            b.wait_deps(b_load)
            b_write = b.transfer_tiles(sB, tc_dsw_b, unroll_axes=(n, k_tile), data=b_load)

        # No s_barrier: each wave's LDS slot is private; load -> write -> read
        # is fully ordered within the wave by the write/read tokens.
        with b.stage(STG_A_READ):
            b.wait_deps(a_write)
            a_frags = b.transfer_tiles(sA, tc_dsr_a, unroll_axes=(m, k_tile))

        with b.stage(STG_B_READ):
            b.wait_deps(b_write)
            b_frags = b.transfer_tiles(sB, tc_dsr_b, unroll_axes=(n, k_tile))

        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_frags)
            for fi, ki, mi, ni in enumerate_flat_coords((n_frags, k_t, m_per_wave, n_per_wave)):
                ai = mi * n_per_wave + ni
                a_d = a_frags.data_at((mi, ki, fi))
                b_d = b_frags.data_at((ni, ki, fi))
                accs[ai] = b.mfma("v_mfma_f32_16x16x16_f16", accs[ai], a_d, b_d)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)
        return accs

    for mi, ni in enumerate_flat_coords((m_per_wave, n_per_wave)):
        ai = mi * n_per_wave + ni
        b.store_multi_fragment_to_global(
            accs_final[ai],
            c_ptr,
            b.slice(TC, {m: mi, n: ni}).offset,
            GLOBAL_STORE_TILE_C,
            GLOBAL_STORE_SUB_TILE_C,
            b.global_store_dword,
        )
    return b.build()


class TestPythonGEMMMultiwave:
    #   - m_t % wpw_m == 0, n_t % wpw_n == 0 (compute split divisibility)
    #   - nw * (m_per_wave + n_per_wave) * k_t * 1024 * pipeline_depth <= 64 KB
    @pytest.mark.parametrize(
        "num_waves_per_wg, num_tiles_per_wg",
        [
            # 1-wave: simple non-square / deep K.
            ((1, 1), (1, 1, 1)),  # baseline
            ((1, 1), (2, 2, 1)),
            ((1, 1), (3, 2, 1)),  # 1w_3x2
            ((1, 1), (6, 6, 1)),  # bigger single-wave tile
            ((1, 1), (2, 2, 3)),  # 1w_2x2_k3 (deep K)
            # 2-wave M-split (2x1).
            ((2, 1), (2, 2, 1)),
            ((2, 1), (6, 2, 1)),  # 2w_6x2
            ((2, 1), (6, 4, 1)),  # 2w_6x4
            # 2-wave N-split.
            ((1, 2), (2, 2, 1)),
            ((1, 2), (2, 6, 1)),
            # 4-wave (2x2): bounded by LDS.
            ((2, 2), (2, 2, 1)),
            ((2, 2), (4, 2, 1)),
            ((2, 2), (4, 4, 1)),
            # 4-wave (4x1): M-heavy split (4w style with smaller tile).
            ((4, 1), (4, 2, 1)),
            ((4, 1), (8, 2, 1)),
        ],
        ids=[
            "1w_1x1x1",
            "1w_2x2x1",
            "1w_3x2x1",
            "1w_6x6x1",
            "1w_2x2x3",
            "2w_2x2x1",
            "2w_6x2x1",
            "2w_6x4x1",
            "1x2w_2x2x1",
            "1x2w_2x6x1",
            "4w_2x2_2x2x1",
            "4w_2x2_4x2x1",
            "4w_2x2_4x4x1",
            "4w_4x1_4x2x1",
            "4w_4x1_8x2x1",
        ],
    )
    @pytest.mark.parametrize(
        "num_workgroups",
        [(1, 1), (2, 2)],
        ids=["wg1x1", "wg2x2"],
    )
    @pytest.mark.parametrize("K", [384])
    def test_gemm_multiwave_python(self, num_workgroups, num_waves_per_wg, num_tiles_per_wg, K):
        wg_m_count, wg_n_count = num_workgroups
        wpw_m, wpw_n = num_waves_per_wg
        m_t, n_t, k_t = num_tiles_per_wg
        if m_t % wpw_m != 0 or n_t % wpw_n != 0:
            pytest.skip(f"m_t={m_t} or n_t={n_t} not divisible by wpw={num_waves_per_wg}")
        nw = wpw_m * wpw_n
        num_threads = nw * WAVE_SIZE
        M_total = wg_m_count * m_t * MFMA_M
        N_total = wg_n_count * n_t * MFMA_N
        stride_a = K * ELT_BYTES
        stride_b = K * ELT_BYTES

        np.random.seed(42 + K + M_total + N_total)
        A = (np.random.randn(M_total, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N_total, K) * 0.1).astype(np.float16)
        C_output = np.zeros(M_total * N_total, dtype=np.float32)

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm_pipelined(num_workgroups, num_waves_per_wg, num_tiles_per_wg, K, stride_a, stride_b)
            asm = compile_mlir_module_to_asm(module, pass_pipeline=make_default_pass_pipeline(PipelineConfig()))

        path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
        if path is None:
            pytest.skip(f"LLVM assembler not compiled with {MCPU} support (unknown target)")

        with hsaco_file(path):
            if not system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"{MCPU} GPU not available")
            execute_hsaco(
                hsaco_path=path,
                kernel_name="gemm_pipelined_multiwave",
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B.flatten()),
                    OutputArray(C_output),
                ],
                grid_dim=(wg_m_count * wg_n_count, 1, 1),
                block_dim=(num_threads, 1, 1),
            )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--wg_m", type=int, default=1)
    parser.add_argument("--wg_n", type=int, default=1)
    parser.add_argument("--wpw_m", type=int, default=2)
    parser.add_argument("--wpw_n", type=int, default=2)
    parser.add_argument("--m_t", type=int, default=2)
    parser.add_argument("--n_t", type=int, default=2)
    parser.add_argument("--k_t", type=int, default=1)
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    num_workgroups = (args.wg_m, args.wg_n)
    num_waves_per_wg = (args.wpw_m, args.wpw_n)
    num_tiles_per_wg = (args.m_t, args.n_t, args.k_t)
    stride_a = args.K * ELT_BYTES
    stride_b = args.K * ELT_BYTES
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_pipelined(num_workgroups, num_waves_per_wg, num_tiles_per_wg, args.K, stride_a, stride_b)
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
