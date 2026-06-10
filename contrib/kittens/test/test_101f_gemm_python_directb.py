"""Pipelined GEMM with shared LDS for A + direct preshuffled B (no LDS for B).

Hill-climb from test_101e: A shared WG-level LDS via clamped 2D cooperative load.
B direct preshuffle load.
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

from kittens_helpers import shuffle_weight


# Compute / WG axes (same as test_101c/d/e).
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# A's cooperative-load inner axes (same as test_101e).
m_load_a = Symbol("m_load_a")
k_load_a = Symbol("k_load_a")
# B's direct-load per-wave per-iter tile axes (no cooperation -- each
# wave loads its own n_per_wave x k_t tiles).
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

# Direct-B preshuffle constants. Each (n_tile, k_tile) of preshuffled B is
# laid out as WAVE_SIZE lanes x 16 bytes (one vx4 per lane). The bytes
# within a tile match the MFMA 16x16x16 register layout, so a direct
# global load gives the right fragments without going through LDS.
PRESHUFFLE_LANE_STRIDE = 16  # vx4 = 16 bytes per lane
PRESHUFFLE_K_TILE_BYTES = WAVE_SIZE * PRESHUFFLE_LANE_STRIDE  # 1024 bytes per (n,k) tile

STG_A_LOAD = 0
STG_B_LOAD = 1
STG_A_WRITE = 2
STG_A_READ = 2
STG_COMPUTE = 3
N_STAGES = STG_COMPUTE + 1


def _build_gemm_pipelined(num_workgroups, num_waves_per_wg, num_tiles_per_wg, K, stride_a, stride_b):
    """Build a multi-WG, multi-wave pipelined GEMM with shared LDS A + direct preshuffled B."""
    wg_m_count, wg_n_count = num_workgroups
    wpw_m, wpw_n = num_waves_per_wg
    m_t, n_t, k_t = num_tiles_per_wg
    nw = wpw_m * wpw_n
    num_threads = nw * WAVE_SIZE

    assert m_t % wpw_m == 0, f"m_t={m_t} not divisible by wpw_m={wpw_m}"
    assert n_t % wpw_n == 0, f"n_t={n_t} not divisible by wpw_n={wpw_n}"
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    lds_total_a = m_t * k_t * TILE_BYTES_A
    # No LDS for B (direct preshuffled load).

    k_step = k_t * TILE_K_ELEMS
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, (
        f"k_iters={k_iters} < N_STAGES={N_STAGES}; pipeliner needs at least N_STAGES iters for prologue+steady-state"
    )
    assert K % TILE_K_ELEMS == 0, f"K={K} not multiple of TILE_K_ELEMS={TILE_K_ELEMS}"
    k_blocks = K // TILE_K_ELEMS  # K-tiles in the preshuffled layout.

    # Bytes-per-N-tile in preshuffled B = all K-tiles for that N strip.
    PRESHUFFLE_N_TILE_BYTES = k_blocks * PRESHUFFLE_K_TILE_BYTES

    M_total = wg_m_count * m_t * MFMA_M
    N_total = wg_n_count * n_t * MFMA_N
    stride_c = N_total * 4

    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

    b = KernelBuilder("gemm_pipe_mod", "gemm_pipelined_directb", target=MCPU)
    b.set_grid_dims(wg_m_count * wg_n_count)
    b.set_block_dims(num_threads)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # A: same TiledCopies as test_101e (LDS path with swizzle).
    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((MFMA_M, 4), (stride_a, 16)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((MFMA_M, 4), (64, 16)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, 8),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((4, MFMA_M), (8, 64)),
        value_layout=Layout(TILE_K_ELEMS // MFMA_K, MFMA_K * ELT_BYTES),
        swizzle=LDS_SWIZZLE,
    )

    # B: direct global load. All 64 lanes load a contiguous block of 16 bytes
    # each within the (1024-byte) preshuffled tile -- no swizzle, no LDS.
    tc_load_b_direct = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout(WAVE_SIZE, PRESHUFFLE_LANE_STRIDE),
        value_layout=Layout(1, 0),
    )
    # C store: MFMA 16x16 accumulator -> row-major fp32 C tile. The lane owns
    # col = lane%16 and the 4-row block starting at (lane//16)*4; value v is
    # the row within that block (4 fp32 D-registers per lane).
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((4, MFMA_N), (4 * stride_c, 4)),
        value_layout=Layout(4, stride_c),
    )

    # Global A layout: same hierarchical shape as test_101e.
    A_TILED = tile(
        Layout((M_total, K), (stride_a, ELT_BYTES)),
        tile_sizes=((MFMA_M, m_t), (TILE_K_ELEMS, k_t)),
        axes=((m, wg_m), (k_tile, global_k)),
    )
    # Global B preshuffle layout.
    B_TILED = Layout(
        sizes=(n_per_wave, wpw_n, wg_n_count, k_t, k_iters),
        strides=(
            PRESHUFFLE_N_TILE_BYTES,  # n_load_b
            n_per_wave * PRESHUFFLE_N_TILE_BYTES,  # wave_n
            n_t * PRESHUFFLE_N_TILE_BYTES,  # wg_n
            PRESHUFFLE_K_TILE_BYTES,  # k_load_b
            k_t * PRESHUFFLE_K_TILE_BYTES,  # global_k
        ),
        axes=(n_load_b, wave_n, wg_n, k_load_b, global_k),
    )
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c, 4)),
        tile_sizes=((MFMA_M, m_per_wave, wpw_m), (MFMA_N, n_per_wave, wpw_n)),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

    # Shared WG-level LDS for A (test_101e's read view).
    LDS_A_READ_TILED = tile(
        Layout((m_t, k_t * TILE_BYTES_A), (k_t * TILE_BYTES_A, 1)),
        tile_sizes=((1, m_per_wave), TILE_BYTES_A),
        axes=((m, wave_m), k_tile),
    )

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

    # Per-WG / per-wave tensors. TA pre-sliced by wg_m (cooperative load
    # in K-loop); TB pre-sliced by both wg_n AND wave_n (each wave does
    # its OWN load -- no cooperation); TC pre-sliced by all 4 axes.
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {wg_n: wg_n_idx, wave_n: wave_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {wg_m: wg_m_idx, wg_n: wg_n_idx, wave_m: wave_m_idx, wave_n: wave_n_idx},
    )

    n_accs = m_per_wave * n_per_wave
    n_frags = TILE_K_ELEMS // MFMA_K  # = 2 for MFMA 16x16x16 with TILE_K_ELEMS=32
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
            # Direct preshuffled B: each lane in each wave loads its vx4 per
            # (n_load_b, k_load_b) tile. No LDS allocation, no s_barrier.
            tb_iter = b.slice(TB, {global_k: k_iv})
            b_vx4 = b.transfer_tiles(tb_iter, tc_load_b_direct, unroll_axes=(n_load_b, k_load_b))

        with b.stage(STG_A_WRITE):
            b.wait_deps(a_load)
            sA_write = Tensor(sA_full.ptr, plan_a.lds_wave_off, plan_a.lds_layout)
            a_write = b.transfer_tiles(sA_write, tc_dsw_a, unroll_axes=plan_a.unroll_axes, data=a_load)

        with b.stage(STG_A_READ):
            b.wait_deps(a_write)
            b.s_barrier()
            sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
            a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile))

        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_vx4)
            # Split each per-tile B vx4 into two vx2 MFMA fragments. The
            # compiler CSE's repeated splits at the same (ni, ki) across
            # the MFMA loop's two fi iterations.
            b_frags_split = {}
            for ni, ki in enumerate_flat_coords((n_per_wave, k_t)):
                b_lo, b_hi = b.split_register_range(b_vx4.data_at((ni, ki)), 2)
                b_frags_split[(ni, ki, 0)] = b_lo
                b_frags_split[(ni, ki, 1)] = b_hi

            for fi, ki, mi, ni in enumerate_flat_coords((n_frags, k_t, m_per_wave, n_per_wave)):
                ai = mi * n_per_wave + ni
                a_d = a_frags.data_at((mi, ki, fi))
                b_d = b_frags_split[(ni, ki, fi)]
                accs[ai] = b.mfma("v_mfma_f32_16x16x16_f16", accs[ai], a_d, b_d)
            b.dealloc_lds(lds_a_h)
        return accs

    # C store: one tiled copy per (m, n) tile, mirroring the load path.
    accs_bundle = LayoutValues.from_flat(Layout((m_per_wave, n_per_wave)), payloads=tuple(accs_final))
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(m, n), data=accs_bundle)
    return b.build()


class TestPythonGEMMDirectB:
    @pytest.mark.parametrize(
        "num_waves_per_wg, num_tiles_per_wg",
        [
            ((2, 1), (2, 2, 1)),
            ((2, 1), (6, 2, 1)),
            ((2, 1), (6, 4, 1)),
            ((1, 2), (2, 6, 1)),
            ((2, 2), (2, 2, 1)),
            ((2, 2), (4, 2, 1)),
            ((2, 2), (4, 4, 1)),
            ((4, 1), (8, 2, 1)),
            ((2, 2), (6, 4, 1)),  # "4w_2x2_6x4"
            ((2, 2), (8, 4, 1)),
            ((4, 2), (4, 2, 1)),
            ((4, 2), (8, 4, 1)),
            ((4, 2), (8, 6, 1)),
            ((4, 2), (12, 4, 1)),
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
            "4w_2x2_6x4x1",
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
    @pytest.mark.parametrize("K", [384])
    def test_gemm_directb_python(self, num_workgroups, num_waves_per_wg, num_tiles_per_wg, K):
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
        stride_b = K * ELT_BYTES  # unused by direct-B kernel; preshuffled B is byte-laid-out per PRESHUFFLE_* constants

        np.random.seed(42 + K + M_total + N_total)
        A = (np.random.randn(M_total, K) * 0.1).astype(np.float16)
        B = (np.random.randn(N_total, K) * 0.1).astype(np.float16)
        B_preshuffled = shuffle_weight(B)
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
                kernel_name="gemm_pipelined_directb",
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B_preshuffled.flatten()),
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
