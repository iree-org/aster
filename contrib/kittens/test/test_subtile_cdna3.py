"""Single 128x128x128xf16 tile on a single workgroup with 4 waves."""

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


# Per-wave compute axes.
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
KERNEL_NAME = "gemm_subtile"
WAVE_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
TILE_K_ELEMS = 32
ELT_BYTES = 2
TILE_BYTES_A = MFMA_M * TILE_K_ELEMS * ELT_BYTES  # 1024
TILE_BYTES_B = MFMA_N * TILE_K_ELEMS * ELT_BYTES  # 1024

# Fixed single config: 1 WG, 4 waves (2x2), one 128x128 C tile.
NUM_WORKGROUPS = (1, 1)
NUM_WAVES_PER_WG = (2, 2)
NUM_TILES_PER_WG = (8, 8, 1)
K_DIM = 128

STG_A_LOAD = 0
STG_B_LOAD = 1
STG_B_WRITE = 1
STG_A_WRITE = 2
STG_A_READ = 2
STG_B_READ = 2
STG_COMPUTE = 3
N_STAGES = STG_COMPUTE + 1


def _build_gemm(num_workgroups, num_waves_per_wg, num_tiles_per_wg, K, stride_a, stride_b):
    """Build the fixed-config GEMM: shared WG LDS for A and B, coop load, s_barrier."""
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
    lds_total_b = n_t * k_t * TILE_BYTES_B

    k_step = k_t * TILE_K_ELEMS
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, f"k_iters={k_iters} < N_STAGES={N_STAGES}"

    M_total = wg_m_count * m_t * MFMA_M
    N_total = wg_n_count * n_t * MFMA_N
    stride_c = N_total * 4

    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

    b = KernelBuilder("gemm_subtile_mod", KERNEL_NAME, target=MCPU)
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
    # cdna3 16x16x16: one vx2 MFMA fragment per ds_read_b64 (frag_k = MFMA_K).
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
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((4, MFMA_N), (4 * stride_c, 4)),
        value_layout=Layout(4, stride_c),
    )

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
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c, 4)),
        tile_sizes=((MFMA_M, m_per_wave, wpw_m), (MFMA_N, n_per_wave, wpw_n)),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

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
            ta_load = Tensor(a_ptr, b.layout_sum(ta_iter.offset, plan_a.global_wave_off), plan_a.global_layout)
            a_load = b.transfer_tiles(ta_load, tc_load_a, unroll_axes=plan_a.unroll_axes)

        with b.stage(STG_B_LOAD):
            lds_b_h, sB_full = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_READ_TILED)
            tb_iter = b.slice(TB, {global_k: k_iv})
            tb_load = Tensor(b_ptr, b.layout_sum(tb_iter.offset, plan_b.global_wave_off), plan_b.global_layout)
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
            b.wait_deps(a_write)
            b.s_barrier()
            sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
            a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile))

        with b.stage(STG_B_READ):
            b.wait_deps(b_write)
            b.s_barrier()
            sB_read = b.slice(sB_full, {wave_n: wave_n_idx})
            b_frags = b.transfer_tiles(sB_read, tc_dsr_b, unroll_axes=(n, k_tile))

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

    accs_bundle = LayoutValues.from_flat(Layout((m_per_wave, n_per_wave)), payloads=tuple(accs_final))
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(m, n), data=accs_bundle)
    return b.build()


class TestSubtileCDNA3:
    """Single fixed config: 1 WG, 4 waves (2x2), one 128x128x128 f16 tile, CDNA3."""

    def test_correctness(self, print_asm=False, print_ir_after_all=False):
        wg_m_count, wg_n_count = NUM_WORKGROUPS
        wpw_m, wpw_n = NUM_WAVES_PER_WG
        m_t, n_t, k_t = NUM_TILES_PER_WG
        nw = wpw_m * wpw_n
        num_threads = nw * WAVE_SIZE
        M_total = wg_m_count * m_t * MFMA_M
        N_total = wg_n_count * n_t * MFMA_N
        assert (M_total, N_total, K_DIM) == (128, 128, 128)
        assert num_threads == 256
        stride_a = K_DIM * ELT_BYTES
        stride_b = K_DIM * ELT_BYTES

        np.random.seed(42 + M_total + N_total + K_DIM)
        A = (np.random.randn(M_total, K_DIM) * 0.1).astype(np.float16)
        B = (np.random.randn(N_total, K_DIM) * 0.1).astype(np.float16)
        C_output = np.zeros(M_total * N_total, dtype=np.float32)

        print_opts = None
        if print_asm or print_ir_after_all:
            from aster.compiler.core import PrintOptions

            print_opts = PrintOptions.from_flags(
                print_ir_after_all=print_ir_after_all,
                print_asm=print_asm,
            )

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            module = _build_gemm(NUM_WORKGROUPS, NUM_WAVES_PER_WG, NUM_TILES_PER_WG, K_DIM, stride_a, stride_b)
            asm = compile_mlir_module_to_asm(
                module,
                pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
                print_opts=print_opts,
            )
        if print_asm:
            print(asm)

        path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
        if path is None:
            pytest.skip(f"LLVM assembler not compiled with {MCPU} support (unknown target)")
        assert os.path.getsize(path) > 0

        with hsaco_file(path):
            if not system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"{MCPU} GPU not available")
            execute_hsaco(
                hsaco_path=path,
                kernel_name=KERNEL_NAME,
                arguments=[InputArray(A.flatten()), InputArray(B.flatten()), OutputArray(C_output)],
                grid_dim=(wg_m_count * wg_n_count, 1, 1),
                block_dim=(num_threads, 1, 1),
            )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    a = parser.parse_args()
    TestSubtileCDNA3().test_correctness(
        print_asm=a.print_asm,
        print_ir_after_all=a.print_ir_after_all,
    )
