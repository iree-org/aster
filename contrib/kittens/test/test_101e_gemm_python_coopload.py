"""Pipelined GEMM with shared WG-level LDS + clamped 2D cooperative load.

Hill-climb from test_101d (multi-wave multi-WG): private per-wave
global->LDS.
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
from coop import make_coop_load_plan


# Per-wave compute axes (same as test_101c/d).
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
# WG-level axes (bound from linear_block_id()).
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
# Per-wave COMPUTE axes (which wave owns which output sub-tile).
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# Cooperative LOAD inner-iter axes (rank-2 per-wave view).
m_load_a = Symbol("m_load_a")
k_load_a = Symbol("k_load_a")
n_load_b = Symbol("n_load_b")
k_load_b = Symbol("k_load_b")

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


def _build_gemm_pipelined(
    num_workgroups,
    num_waves_per_wg,
    num_tiles_per_wg,
    K,
    stride_a,
    stride_b,
    use_conservative_barriers=False,
):
    """Build a multi-WG, multi-wave pipelined GEMM with shared LDS + cooperative load."""
    wg_m_count, wg_n_count = num_workgroups
    wpw_m, wpw_n = num_waves_per_wg
    m_t, n_t, k_t = num_tiles_per_wg  # per-WG (test_101d convention)
    nw = wpw_m * wpw_n
    num_threads = nw * WAVE_SIZE

    assert m_t % wpw_m == 0, f"m_t={m_t} not divisible by wpw_m={wpw_m}"
    assert n_t % wpw_n == 0, f"n_t={n_t} not divisible by wpw_n={wpw_n}"
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    # Shared per-WG LDS.
    lds_total_a = m_t * k_t * TILE_BYTES_A
    lds_total_b = n_t * k_t * TILE_BYTES_B

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

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined_coopload", target=MCPU)
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
    # C store: MFMA 16x16 accumulator -> row-major fp32 C tile. The lane owns
    # col = lane%16 and the 4-row block starting at (lane//16)*4; value v is
    # the row within that block (4 fp32 D-registers per lane).
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((4, MFMA_N), (4 * stride_c, 4)),
        value_layout=Layout(4, stride_c),
    )

    # Global A/B layouts: hierarchical (MFMA-tile, m_t) x (TILE_K_ELEMS, k_t).
    # Note: the tile is like test_101d, without the wave_m/wave_n component and
    # we instead b.slice(TA, {wg_m: wg_m_idx, global_k: k_iv}).
    A_TILED = tile(
        Layout((M_total, K), (stride_a, ELT_BYTES)),
        tile_sizes=((MFMA_M, m_t), (TILE_K_ELEMS, k_t)),
        axes=((m, wg_m), (k_tile, global_k)),
    )
    B_TILED = tile(
        Layout((N_total, K), (stride_b, ELT_BYTES)),
        tile_sizes=((MFMA_N, n_t), (TILE_K_ELEMS, k_t)),
        axes=((n, wg_n), (k_tile, global_k)),
    )
    # C compute-distributed (same as test_101d).
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c, 4)),
        tile_sizes=((MFMA_M, m_per_wave, wpw_m), (MFMA_N, n_per_wave, wpw_n)),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

    # Shared WG-level LDS read views (sliced per wave by wave_m / wave_n,
    # same as test_101d but over the full per-WG (m_t, n_t) tile -- shared
    # across all waves rather than partitioned per linear wave id).
    LDS_A_READ_TILED = tile(
        Layout((m_t, k_t * TILE_BYTES_A), (k_t * TILE_BYTES_A, 1)),
        tile_sizes=((1, m_per_wave), TILE_BYTES_A),
        axes=((m, wave_m), k_tile),
    )
    LDS_B_READ_TILED = tile(
        Layout((n_t, k_t * TILE_BYTES_B), (k_t * TILE_BYTES_B, 1)),
        tile_sizes=((1, n_per_wave), TILE_BYTES_B),
        axes=((n, wave_n), k_tile),
    )

    # Per-WG and per-wave-compute distributions; cooperative-load plans for A and B.
    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg_m_count, wg_n_count))
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    plan_a = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((m_t, k_t), (MFMA_M * stride_a, TILE_K_ELEMS * ELT_BYTES)),
        wg_tile_lds=Layout((m_t, k_t), (k_t * TILE_BYTES_A, TILE_BYTES_A)),
        spatial_axis=m_load_a,
        k_axis=k_load_a,
    )
    plan_b = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((n_t, k_t), (MFMA_N * stride_b, TILE_K_ELEMS * ELT_BYTES)),
        wg_tile_lds=Layout((n_t, k_t), (k_t * TILE_BYTES_B, TILE_BYTES_B)),
        spatial_axis=n_load_b,
        k_axis=k_load_b,
    )

    # Same slicing form as test_101d: without a wave_m / wave_n binding here.
    # The load is cooperative rather than wave-distributed, offset lives in
    # plan_a/b.
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {wg_n: wg_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {wg_m: wg_m_idx, wg_n: wg_n_idx, wave_m: wave_m_idx, wave_n: wave_n_idx},
    )

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
            lds_a_h, sA_full = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_READ_TILED)
            ta_iter = b.slice(TA, {global_k: k_iv})
            ta_load = Tensor(
                a_ptr,
                b.layout_sum(ta_iter.offset, plan_a.global_wave_off),
                plan_a.global_layout,
            )
            a_load = b.transfer_tiles(ta_load, tc_load_a, unroll_axes=plan_a.unroll_axes)

        with b.stage(STG_B_LOAD):
            lds_b_h, sB_full = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_READ_TILED)
            tb_iter = b.slice(TB, {global_k: k_iv})
            tb_load = Tensor(
                b_ptr,
                b.layout_sum(tb_iter.offset, plan_b.global_wave_off),
                plan_b.global_layout,
            )
            b_load = b.transfer_tiles(tb_load, tc_load_b, unroll_axes=plan_b.unroll_axes)

        with b.stage(STG_A_WRITE):
            b.wait_deps(a_load)
            sA_write = Tensor(sA_full.ptr, plan_a.lds_wave_off, plan_a.lds_layout)
            a_write = b.transfer_tiles(sA_write, tc_dsw_a, unroll_axes=plan_a.unroll_axes, data=a_load)

        with b.stage(STG_B_WRITE):
            b.wait_deps(b_load)
            sB_write = Tensor(sB_full.ptr, plan_b.lds_wave_off, plan_b.lds_layout)
            b_write = b.transfer_tiles(sB_write, tc_dsw_b, unroll_axes=plan_b.unroll_axes, data=b_load)

        with b.stage(STG_A_READ):
            if use_conservative_barriers:
                b.wait_deps(a_write)
                b.barrier()
                sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
                a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile))
            else:
                wfence_a = b.wait_deps(a_write)
                bfence_a = b.token_barrier(wfence_a)
                sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
                a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile), fence_token=bfence_a)

        with b.stage(STG_B_READ):
            if use_conservative_barriers:
                b.wait_deps(b_write)
                b.barrier()
                sB_read = b.slice(sB_full, {wave_n: wave_n_idx})
                b_frags = b.transfer_tiles(sB_read, tc_dsr_b, unroll_axes=(n, k_tile))
            else:
                wfence_b = b.wait_deps(b_write)
                bfence_b = b.token_barrier(wfence_b)
                sB_read = b.slice(sB_full, {wave_n: wave_n_idx})
                b_frags = b.transfer_tiles(sB_read, tc_dsr_b, unroll_axes=(n, k_tile), fence_token=bfence_b)

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

    # C store: one tiled copy per (m, n) tile, mirroring the load path.
    accs_bundle = LayoutValues.from_flat(Layout((m_per_wave, n_per_wave)), payloads=tuple(accs_final))
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(m, n), data=accs_bundle)
    return b.build()


class TestPythonGEMMCoopLoad:
    @pytest.mark.parametrize(
        "num_waves_per_wg, num_tiles_per_wg",
        [
            # Multi-wave that fit without private-LDS budget.
            ((2, 1), (2, 2, 1)),
            ((2, 1), (6, 2, 1)),  # 2w_6x2
            ((2, 1), (6, 4, 1)),  # 2w_6x4
            ((1, 2), (2, 6, 1)),
            ((2, 2), (2, 2, 1)),
            ((2, 2), (4, 2, 1)),
            ((2, 2), (4, 4, 1)),
            ((4, 1), (8, 2, 1)),
            # Configs only possible with shared LDS / coop load.
            ((2, 2), (6, 4, 1)),  # 4w_2x2_6x4 (private LDS 72 KB; shared 36 KB)
            ((2, 2), (8, 4, 1)),  # 4-wave bigger
            ((4, 2), (4, 2, 1)),  # 8-wave small
            ((4, 2), (8, 4, 1)),  # 8-wave bigger
            ((4, 2), (8, 6, 1)),  # 8-wave near-budget
            ((4, 2), (12, 4, 1)),  # 8-wave M-heavy
        ],
        ids=[
            "2w_2x2x1",
            "2w_6x2x1",
            "2w_6x4x1",
            "1x2w_2x6x1",
            "4w_2x2_2x2x1",
            "4w_2x2_4x2x1",
            "4w_2x2_4x4x1",
            "4w_4x1_8x2x1",
            "4w_2x2_6x4x1",  # newly fits (vs test_101d)
            "4w_2x2_8x4x1",
            "8w_4x2_4x2x1",
            "8w_4x2_8x4x1",
            "8w_4x2_8x6x1",
            "8w_4x2_12x4x1",
        ],
    )
    @pytest.mark.parametrize(
        "num_workgroups",
        [(1, 1), (2, 2)],
        ids=["wg1x1", "wg2x2"],
    )
    @pytest.mark.parametrize(
        "use_conservative_barriers",
        [False, True],
        ids=["token_barrier", "s_barrier"],
    )
    @pytest.mark.parametrize("K", [384])
    def test_gemm_coopload_python(
        self, num_workgroups, num_waves_per_wg, num_tiles_per_wg, K, use_conservative_barriers
    ):
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
            module = _build_gemm_pipelined(
                num_workgroups,
                num_waves_per_wg,
                num_tiles_per_wg,
                K,
                stride_a,
                stride_b,
                use_conservative_barriers=use_conservative_barriers,
            )
            asm = compile_mlir_module_to_asm(module, pass_pipeline=make_default_pass_pipeline(PipelineConfig()))

        path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
        if path is None:
            pytest.skip(f"LLVM assembler not compiled with {MCPU} support (unknown target)")

        with hsaco_file(path):
            if not system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"{MCPU} GPU not available")
            execute_hsaco(
                hsaco_path=path,
                kernel_name="gemm_pipelined_coopload",
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
    parser.add_argument("--K", type=int, default=384)
    parser.add_argument("--wg_m", type=int, default=1)
    parser.add_argument("--wg_n", type=int, default=1)
    parser.add_argument("--wpw_m", type=int, default=2)
    parser.add_argument("--wpw_n", type=int, default=2)
    parser.add_argument("--m_t", type=int, default=4)
    parser.add_argument("--n_t", type=int, default=4)
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
