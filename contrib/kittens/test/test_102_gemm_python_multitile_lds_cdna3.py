"""CDNA3 (MI300, gfx942) multi-tile GEMM, LDS operand path, layout-driven API.

Both A and B flow through LDS via cooperative flat loads (global_load_dwordx4
-> ds_write_b64 -> ds_read_b64 vx2 fragments -> v_mfma_f32_16x16x16_f16).
Divisible M/N/K, single K-loop (no persistence), flat loads/store (no buffer OOB).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np
import pytest

from aster import ir
import tempfile

from aster.layout import (
    Layout,
    LayoutValues,
    Swizzle,
    Symbol,
    Tensor,
    enumerate_flat_coords,
    tile,
)
from aster.dialects.kernel_builder_with_layouts import (
    global_load_dwordx4,
    global_store_dword,
    ds_read_64b,
    ds_write_64b,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_gpu
from aster.pass_pipelines import make_default_pass_pipeline

from coop import make_coop_load_plan

from kittens.gemm_config import (
    A as OP_A,
    B as OP_B,
    C as OP_C,
    DIM_M,
    DIM_N,
    DIM_K,
    GemmSpec,
    GemmMappingSpec,
    OperandPath,
    WeakScaledMappedGemmInstance,
)


# Compute / WG axes.
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# A's / B's cooperative-load inner axes.
m_load_a = Symbol("m_load_a")
k_load_a = Symbol("k_load_a")
n_load_b = Symbol("n_load_b")
k_load_b = Symbol("k_load_b")


def _build_multitile_gemm(cfg: "MultitileGemmInstance") -> ir.Module:
    """Build a multi-tile multi-wave multi-WG pipelined GEMM kernel (LDS path).

    Both A and B flow through LDS via flat global loads (cooperative
    global_load_dwordx4 -> ds_write_b64 -> ds_read_b64 vx2 fragments ->
    v_mfma_f32_16x16x16_f16). cdna3 (gfx942): each ds_read_b64 yields a
    vx2 fragment that is the MFMA operand directly (no vx4 join).
    Divisible M/N/K, single K-loop (no persistence), flat loads/store
    (no buffer OOB).
    """

    from kittens_helpers import PIPELINE_STRATEGIES
    from aster._mlir_libs._amdgcn import AGPRRangeType

    spec, mapping = cfg.spec, cfg.mapping
    gs = spec.gemm_size
    wg = mapping.num_workgroups_per_kernel
    wpw = mapping.num_waves_per_workgroup
    tpw = mapping.num_tiles_per_wave
    twg = mapping.num_tiles_per_workgroup
    ws = mapping.wave_size
    mfma_m, mfma_n, mfma_k = spec.mfma_shape[DIM_M], spec.mfma_shape[DIM_N], spec.mfma_shape[DIM_K]
    elt_bytes = spec.elt_bytes_a

    # Per-transfer-tile geometry (from spec+mapping on cfg).
    tile_k_elems = cfg.transfer_tile_k_elems
    tile_bytes = cfg.transfer_tile_bytes
    # cdna3 (vx2 direct): each ds_read_b64 yields one vx2 fragment that is the
    # MFMA operand directly (no frag_k=mfma_k//2 splitting, no join_vx2_to_vx4).
    frag_k = mfma_k  # K elements per ds_read_b64 vx2 fragment
    n_frags = tile_k_elems // frag_k  # vx2 fragments per K-tile

    m_t, n_t, k_t = tpw[DIM_M], tpw[DIM_N], tpw[DIM_K]
    twg_m, twg_n = twg[DIM_M], twg[DIM_N]
    nw = mapping.num_waves
    num_threads = mapping.num_threads

    assert twg_m == wpw[DIM_M] * m_t, f"twg_m({twg_m}) != wpw_m({wpw[DIM_M]}) * m_t({m_t})"
    assert twg_n == wpw[DIM_N] * n_t, f"twg_n({twg_n}) != wpw_n({wpw[DIM_N]}) * n_t({n_t})"
    m_per_wave, n_per_wave = m_t, n_t

    # Operand strides (row-major M x K / N x K / M x N).
    ol_a, ol_b, ol_c = spec.operand_layout(OP_A), spec.operand_layout(OP_B), spec.operand_layout(OP_C)
    stride_a, stride_b = ol_a.strides[0], ol_b.strides[0]
    stride_c_row, stride_c_col = ol_c.strides[0], ol_c.strides[1]

    k_step = k_t * tile_k_elems
    assert gs[DIM_K] % k_step == 0, f"K={gs[DIM_K]} must be divisible by k_t*tile_k_elems={k_step}"
    k_iters = gs[DIM_K] // k_step
    n_accs = m_per_wave * n_per_wave

    # LDS per-WG byte sizes (A loads twg_m x k_t tiles, B loads twg_n x k_t).
    lds_total_a = k_t * twg_m * tile_bytes
    lds_total_b = k_t * twg_n * tile_bytes

    # Pipeline stage assignments from strategy. A and B share a stage role each.
    stg = PIPELINE_STRATEGIES[mapping.pipeline_strategy]
    STG_A_LOAD = stg["A_LOAD"]
    STG_A_LDS_WRITE = stg["A_LDS_WRITE"]
    STG_A_LDS_READ = stg["A_LDS_READ"]
    STG_B_LOAD = stg["B_LOAD"]
    STG_B_LDS_WRITE = stg["B_LDS_WRITE"]
    STG_B_LDS_READ = stg["B_LDS_READ"]
    STG_COMPUTE = stg["COMPUTE"]

    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=4)

    M_total = wg[DIM_M] * twg_m * mfma_m
    N_total = wg[DIM_N] * twg_n * mfma_n

    b = KernelBuilder("gemm_mod", cfg.kernel_name, target=mapping.mcpu)
    b.set_block_dims(num_threads)
    b.set_grid_dims(mapping.num_workgroups)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    ax4_type = AGPRRangeType.get(b._ctx, size=4)

    # -- Tiled copies (lane-partitioned hardware transfers) --
    # A: flat global load -> LDS write (swizzled) -> LDS read (vx2 fragments).
    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((mfma_m, ws // mfma_m), (stride_a, mapping.global_load_bytes)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((mfma_m, ws // mfma_m), (ws, mapping.global_load_bytes)),
        value_layout=Layout(mapping.global_load_bytes // mapping.ds_write_bytes, mapping.ds_write_bytes),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((ws // mfma_m, mfma_m), (mapping.ds_read_bytes, ws)),
        value_layout=Layout(n_frags, frag_k * elt_bytes),
        swizzle=LDS_SWIZZLE,
    )
    # B mirrors A with n in place of m.
    tc_load_b = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((mfma_n, ws // mfma_n), (stride_b, mapping.global_load_bytes)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_b = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((mfma_n, ws // mfma_n), (ws, mapping.global_load_bytes)),
        value_layout=Layout(mapping.global_load_bytes // mapping.ds_write_bytes, mapping.ds_write_bytes),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_b = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((ws // mfma_n, mfma_n), (mapping.ds_read_bytes, ws)),
        value_layout=Layout(n_frags, frag_k * elt_bytes),
        swizzle=LDS_SWIZZLE,
    )
    # C store: MFMA 16x16 accumulator -> row-major fp32 C tile. The lane owns
    # col = lane%mfma_n and the n_agprs-row block starting at (lane//mfma_n)*n_agprs;
    # value v is the row within that block (n_agprs fp32 D-registers per lane).
    n_agprs = ws // mfma_n
    tc_store_c = b.make_tiled_copy_descriptor(
        global_store_dword,
        thread_layout=Layout((n_agprs, mfma_n), (n_agprs * stride_c_row, stride_c_col)),
        value_layout=Layout(n_agprs, stride_c_row),
    )

    # -- Global tiled layouts (byte-addressed; WG block + K-iter outer axes) --
    A_TILED = tile(
        Layout((M_total, gs[DIM_K]), (stride_a, elt_bytes)),
        tile_sizes=(twg_m * mfma_m, k_step),
        axes=(wg_m, global_k),
    )
    B_TILED = tile(
        Layout((N_total, gs[DIM_K]), (stride_b, elt_bytes)),
        tile_sizes=(twg_n * mfma_n, k_step),
        axes=(wg_n, global_k),
    )
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c_row, stride_c_col)),
        tile_sizes=((mfma_m, m_per_wave, wpw[DIM_M]), (mfma_n, n_per_wave, wpw[DIM_N])),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

    # Shared WG-level LDS read views (A keyed by (m, wave_m), B by (n, wave_n)).
    LDS_A_READ_TILED = tile(
        Layout((twg_m, k_t * tile_bytes), (k_t * tile_bytes, 1)),
        tile_sizes=((1, m_per_wave), tile_bytes),
        axes=((m, wave_m), k_tile),
    )
    LDS_B_READ_TILED = tile(
        Layout((twg_n, k_t * tile_bytes), (k_t * tile_bytes, 1)),
        tile_sizes=((1, n_per_wave), tile_bytes),
        axes=((n, wave_n), k_tile),
    )

    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg[DIM_M], wg[DIM_N]))
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw[DIM_M], wpw[DIM_N]))

    plan_a = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((twg_m, k_t), (mfma_m * stride_a, tile_k_elems * elt_bytes)),
        wg_tile_lds=Layout((twg_m, k_t), (k_t * tile_bytes, tile_bytes)),
        spatial_axis=m_load_a,
        k_axis=k_load_a,
    )
    plan_b = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((twg_n, k_t), (mfma_n * stride_b, tile_k_elems * elt_bytes)),
        wg_tile_lds=Layout((twg_n, k_t), (k_t * tile_bytes, tile_bytes)),
        spatial_axis=n_load_b,
        k_axis=k_load_b,
    )

    # Per-WG tensors sliced by wg_m / wg_n (cooperative load inside K-loop).
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {wg_n: wg_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {wg_m: wg_m_idx, wg_n: wg_n_idx, wave_m: wave_m_idx, wave_n: wave_n_idx},
    )

    # -- Accumulator buffer --
    c_buf = b.memref_alloca(b.constant_index(n_accs), ax4_type)

    @b.foreach_tile(n_accs)
    def _(idx):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, idx)

    c0, c1 = b.constant_index(0), b.constant_index(1)

    # -- K-loop (void -- accumulators live in c_buf) --
    @b.loop(c0, b.constant_index(k_iters), c1)
    def body(k_iv):
        with b.stage(min(STG_A_LOAD, STG_B_LOAD)):
            lds_a_h, sA_full = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_READ_TILED)
            lds_b_h, sB_full = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_READ_TILED)

        with b.stage(STG_A_LOAD):
            ta_iter = b.slice(TA, {global_k: k_iv})
            ta_load = Tensor(
                a_ptr,
                b.layout_sum(ta_iter.offset, plan_a.global_wave_off),
                plan_a.global_layout,
            )
            a_load = b.transfer_tiles(ta_load, tc_load_a, unroll_axes=plan_a.unroll_axes)

        with b.stage(STG_B_LOAD):
            tb_iter = b.slice(TB, {global_k: k_iv})
            tb_load = Tensor(
                b_ptr,
                b.layout_sum(tb_iter.offset, plan_b.global_wave_off),
                plan_b.global_layout,
            )
            b_load = b.transfer_tiles(tb_load, tc_load_b, unroll_axes=plan_b.unroll_axes)

        with b.stage(STG_A_LDS_WRITE):
            b.wait_deps(a_load)
            sA_write = Tensor(sA_full.ptr, plan_a.lds_wave_off, plan_a.lds_layout)
            a_write = b.transfer_tiles(sA_write, tc_dsw_a, unroll_axes=plan_a.unroll_axes, data=a_load)

        with b.stage(STG_B_LDS_WRITE):
            b.wait_deps(b_load)
            sB_write = Tensor(sB_full.ptr, plan_b.lds_wave_off, plan_b.lds_layout)
            b_write = b.transfer_tiles(sB_write, tc_dsw_b, unroll_axes=plan_b.unroll_axes, data=b_load)

        with b.stage(STG_A_LDS_READ):
            b.wait_deps(a_write)
            b.s_barrier()
            sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
            a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile))
            b.dealloc_lds(lds_a_h)

        with b.stage(STG_B_LDS_READ):
            b.wait_deps(b_write)
            sB_read = b.slice(sB_full, {wave_n: wave_n_idx})
            b_frags = b.transfer_tiles(sB_read, tc_dsr_b, unroll_axes=(n, k_tile))
            b.dealloc_lds(lds_b_h)

        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_frags)
            for fi, ki, mi, ni in enumerate_flat_coords((n_frags, k_t, m_per_wave, n_per_wave)):
                acc_idx = b.constant_index(mi * n_per_wave + ni)
                acc = b.memref_load(c_buf, acc_idx)
                a_d = a_frags.data_at((mi, ki, fi))
                b_d = b_frags.data_at((ni, ki, fi))
                new_acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_d, b_d)
                b.memref_store(new_acc, c_buf, acc_idx)

    # -- Store C tiles -- pass AGPR accs directly (split, no materialize);
    # transfer_tiles splits each acc into n_agprs D-registers.
    accs_bundle = LayoutValues.from_flat(
        Layout((m_per_wave, n_per_wave)),
        payloads=tuple(b.memref_load(c_buf, b.constant_index(i)) for i in range(n_accs)),
    )
    b.transfer_tiles(TC, tc_store_c, unroll_axes=(m, n), data=accs_bundle)

    return b.build()


# ---------------------------------------------------------------------------
# Harness-compatible compile function (reusable by bench sweep / single runner)
# ---------------------------------------------------------------------------

KERNEL_NAME = "gemm_multitile"


class MultitileGemmInstance(WeakScaledMappedGemmInstance):
    """Config with kernel_name override and per-transfer-tile geometry derivations.

    The transfer tile is the per-wave global load granule: each wave reads a
    mfma_m x tile_k_elems block per cooperative load. The geometry is fully
    determined by spec (mfma_shape, elt_bytes_a) + mapping (wave_size,
    global_load_bytes).
    """

    @property
    def kernel_name(self):
        return KERNEL_NAME

    @property
    def transfer_tile_k_elems(self) -> int:
        """K elements per transfer tile: (lanes_per_K_row) * (elements_per_lane)."""
        return (self.mapping.wave_size // self.spec.mfma_shape[DIM_M]) * (
            self.mapping.global_load_bytes // self.spec.elt_bytes_a
        )

    @property
    def transfer_tile_row_bytes(self) -> int:
        """Bytes per row of one transfer tile (= tile_k_elems * elt_bytes_a)."""
        return self.transfer_tile_k_elems * self.spec.elt_bytes_a

    @property
    def transfer_tile_bytes(self) -> int:
        """Total bytes of one transfer tile (= mfma_m * tile_row_bytes)."""
        return self.spec.mfma_shape[DIM_M] * self.transfer_tile_row_bytes


def compile_multitile_gemm(cfg, output_hsaco_path, **kw):
    """Compile a multi-tile GEMM config to HSACO."""
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_multitile_gemm(cfg)
        pipeline = make_default_pass_pipeline(
            cfg.mapping,
            num_vgprs=kw.get("num_vgprs", 256),
            num_agprs=kw.get("num_agprs", 256),
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=pipeline,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=kw.get("print_ir_after_all", False),
                print_asm=kw.get("print_asm", False),
            ),
        )
    path = assemble_to_hsaco(asm, target=cfg.mapping.mcpu, wavefront_size=64, output_path=output_hsaco_path)
    assert path is not None, "assemble_to_hsaco returned None"
    return path, asm


def execute_multitile_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled HSACO.

    LDS path: B flows through LDS (no preshuffle).
    Returns (C_output, times_ns).
    """
    mcpu = cfg.mapping.mcpu
    if not skip_gpu_check and not system_has_gpu(mcpu):
        pytest.skip(f"GPU {mcpu} not available, skip execution")

    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)
    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        arguments=[InputArray(A.flatten()), InputArray(B.flatten()), OutputArray(C_output)],
        grid_dim=(cfg.mapping.num_workgroups, 1, 1),
        block_dim=(cfg.mapping.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


def _make_instance(
    num_workgroups,
    num_waves_per_wg,
    num_tiles_per_wg,
    k_mult,
    pipeline_strategy=1,
    mcpu=None,
    rotate_compute_stage=False,
    ll_sched=0,
    ll_ilp_sched=-1,
):
    """Build a MultitileGemmInstance from list parameters.

    M, N, K are derived from the tile grid, mcpu (which selects the MFMA
    shape), and GemmMappingSpec defaults for transfer widths.
    """
    from kittens.gemm_config import mfma_shape_for_mcpu

    mfma_shape = mfma_shape_for_mcpu(mcpu) if mcpu else None
    assert num_tiles_per_wg[DIM_M] % num_waves_per_wg[DIM_M] == 0, (
        f"twg_m({num_tiles_per_wg[DIM_M]}) not divisible by wpw_m({num_waves_per_wg[DIM_M]})"
    )
    assert num_tiles_per_wg[DIM_N] % num_waves_per_wg[DIM_N] == 0, (
        f"twg_n({num_tiles_per_wg[DIM_N]}) not divisible by wpw_n({num_waves_per_wg[DIM_N]})"
    )
    mapping_kw = {}
    if mcpu is not None:
        mapping_kw["mcpu"] = mcpu
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[
            num_tiles_per_wg[DIM_M] // num_waves_per_wg[DIM_M],
            num_tiles_per_wg[DIM_N] // num_waves_per_wg[DIM_N],
            num_tiles_per_wg[DIM_K],
        ],
        pipeline_strategy=pipeline_strategy,
        operand_path=OperandPath.LDS,
        rotate_compute_stage=rotate_compute_stage,
        ll_sched=ll_sched,
        ll_ilp_sched=ll_ilp_sched,
        **mapping_kw,
    )
    spec_kw = {} if mfma_shape is None else {"mfma_shape": mfma_shape}
    probe_spec = GemmSpec.from_sizes(1, 1, 1, **spec_kw)
    mfma = probe_spec.mfma_shape
    tile_k_elems = (mapping.wave_size // mfma[DIM_M]) * (mapping.global_load_bytes // probe_spec.elt_bytes_a)
    M = num_workgroups[DIM_M] * num_tiles_per_wg[DIM_M] * mfma[DIM_M]
    N = num_workgroups[DIM_N] * num_tiles_per_wg[DIM_N] * mfma[DIM_N]
    k = k_mult * num_tiles_per_wg[DIM_K] * tile_k_elems
    return MultitileGemmInstance(GemmSpec.from_sizes(M, N, k, **spec_kw), mapping)


def _run_multitile(cfg):
    """Compile + run a multi-tile GEMM, verify against numpy."""
    gs = cfg.gemm_size

    np.random.seed(42 + gs[DIM_M] + gs[DIM_N] + gs[DIM_K])
    A_mat = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
    B_mat = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
        compile_multitile_gemm(cfg, tmp.name)
        C_output, _ = execute_multitile_hsaco(cfg, tmp.name, 1, A_mat, B_mat)

    expected = (A_mat.astype(np.float32) @ B_mat.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


def _min_k_iters(twg_k, ps):
    """Minimum K iterations for a given pipeline strategy."""
    from kittens_helpers import PIPELINE_STRATEGIES as PS

    return max(PS[ps].values()) + 1


class TestGeometry:
    """Wave geometry sweep: 1w through 8w, fixed pipeline/operand-path/LDS-stage.

    Validates that different wave counts and tile shapes produce correct results.
    Pipeline, operand path, and LDS stage are tested independently below.
    """

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            # 1 wave
            ([1, 1, 1], [3, 2, 1]),
            ([1, 1, 1], [2, 2, 3]),  # deep K (k_t=3)
            # 2 waves (2x1)
            ([2, 1, 1], [6, 2, 1]),
            # 4 waves (2x2)
            ([2, 2, 1], [8, 4, 1]),
            ([2, 2, 1], [6, 4, 1]),  # non-power-of-2
            ([2, 2, 1], [6, 6, 1]),  # non-power-of-2 both
            # 4 waves (4x1)
            ([4, 1, 1], [8, 7, 1]),
            ([4, 1, 1], [12, 5, 1]),
            # 4 waves (1x4)
            ([1, 4, 1], [10, 8, 1]),
            # 8 waves (4x2)
            ([4, 2, 1], [12, 6, 1]),
            # 8 waves (2x4)
            ([2, 4, 1], [8, 8, 1]),
            ([2, 4, 1], [12, 8, 1]),
        ],
        ids=[
            "1w_3x2",
            "1w_2x2x3_deepK",
            "2w_6x2",
            "4w_2x2_8x4",
            "4w_2x2_6x4_npow2",
            "4w_2x2_6x6_npow2",
            "4w_4x1_8x7",
            "4w_4x1_12x5",
            "4w_1x4_10x8",
            "8w_4x2_12x6",
            "8w_2x4_8x8",
            "8w_2x4_12x8",
        ],
    )
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg):
        cfg = _make_instance([1, 1, 1], num_waves_per_wg, num_tiles_per_wg, k_mult=4, pipeline_strategy=3)
        _run_multitile(cfg)


class TestPipeline:
    """Pipeline strategy x k_mult sweep, fixed geometry.

    Tests pipeline depth interaction with K iterations. Uses two
    representative geometries (small 4w and large 8w).
    """

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([2, 2, 1], [6, 4, 1]),
            ([4, 2, 1], [12, 6, 1]),
        ],
        ids=["4w_2x2_6x4", "8w_4x2_12x6"],
    )
    @pytest.mark.parametrize("k_mult", [2, 4, 8], ids=["km2", "km4", "km8"])
    @pytest.mark.parametrize("pipeline_strategy", [1, 2, 3, 4, 5, 6], ids=["ps1", "ps2", "ps3", "ps4", "ps5", "ps6"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy, rotate_compute_stage):
        k_t = num_tiles_per_wg[DIM_K]
        if k_mult < _min_k_iters(k_t, pipeline_strategy):
            pytest.skip(f"k_mult={k_mult} < min_k_iters for ps{pipeline_strategy}")
        cfg = _make_instance(
            [1, 1, 1],
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_multitile(cfg)


class TestMultiWG:
    """Multi-workgroup correctness, orthogonal to pipeline/operand-path sweep."""

    @pytest.mark.parametrize(
        "num_workgroups,num_waves_per_wg,num_tiles_per_wg",
        [
            ([3, 2, 1], [1, 1, 1], [3, 2, 1]),
            ([2, 2, 1], [2, 2, 1], [4, 4, 1]),
            ([2, 3, 1], [2, 2, 1], [6, 6, 1]),
        ],
        ids=["mwg3x2_1w_3x2", "mwg2x2_4w_4x4", "mwg2x3_4w_6x6"],
    )
    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(
        self, num_workgroups, num_waves_per_wg, num_tiles_per_wg, pipeline_strategy, rotate_compute_stage
    ):
        cfg = _make_instance(
            num_workgroups,
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult=4,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_multitile(cfg)


class TestOperandPath:
    """Operand path (LDS only in this leaf) sweep, orthogonal to geometry/pipeline."""

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([1, 1, 1], [3, 2, 1]),
            ([2, 2, 1], [6, 4, 1]),
            ([4, 2, 1], [12, 6, 1]),
        ],
        ids=["1w_3x2", "4w_6x4", "8w_12x6"],
    )
    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg, pipeline_strategy, rotate_compute_stage):
        cfg = _make_instance(
            [1, 1, 1],
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult=4,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_multitile(cfg)


class TestLLSched:
    @pytest.mark.parametrize("ll_sched", [0, 1, 2, 3, 5], ids=lambda v: f"llsched{v}")
    def test_correctness(self, ll_sched):
        cfg = _make_instance(
            [1, 1, 1],
            [2, 2, 1],
            [6, 4, 1],
            k_mult=4,
            pipeline_strategy=3,
            ll_sched=ll_sched,
        )
        _run_multitile(cfg)


# ---------------------------------------------------------------------------
# Resource estimation accuracy tests
# ---------------------------------------------------------------------------


class TestResourceEstimates:
    """Compile configs and verify LDS/VGPR/AGPR estimates vs actual metadata."""

    _CONFIGS = [
        # (wg, wpw, twg, k_mult, ps, id)
        ([1, 1, 1], [1, 1, 1], [3, 2, 1], 4, 1, "1w_ps1_lds"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 1, "4w_ps1_lds"),
        ([1, 1, 1], [4, 2, 1], [8, 8, 1], 2, 1, "8w_ps1_lds"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 3, "4w_ps3_lds"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 5, "4w_ps5_lds"),
        # -- Asymmetric strategies --
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 2, "4w_ps2_lds"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 4, "4w_ps4_lds"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 6, "4w_ps6_lds"),
    ]

    @pytest.mark.parametrize(
        "wg,wpw,twg,k_mult,ps",
        [(c[0], c[1], c[2], c[3], c[4]) for c in _CONFIGS],
        ids=[c[5] for c in _CONFIGS],
    )
    def test_resource_estimate_accuracy(self, wg, wpw, twg, k_mult, ps):
        """Compile a config and check estimates vs actual metadata.

        Tolerances:
        - LDS:   estimate must be >= actual AND within 15% (tight).
        - VGPRs: within [0.65, 1.5] of actual (structural model, regalloc varies).
        - AGPRs: exact match expected (purely determined by tile shape).
        """
        from aster.compiler.metadata import parse_asm_kernel_resources

        cfg = _make_instance(wg, wpw, twg, k_mult, pipeline_strategy=ps)
        mapping = cfg.mapping

        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            _, asm = compile_multitile_gemm(cfg, tmp.name)

        resources = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME)
        assert KERNEL_NAME in resources, f"kernel {KERNEL_NAME} not found in ASM metadata"
        actual = resources[KERNEL_NAME]

        est_lds = mapping.lds_bytes()
        est_vgprs = mapping.estimated_vgprs()
        est_agprs = mapping.estimated_agprs()

        # LDS: estimate must be >= actual (conservative) and within 15%
        assert est_lds >= actual.lds_bytes, f"LDS estimate {est_lds} < actual {actual.lds_bytes} -- not conservative!"
        if actual.lds_bytes > 0:
            lds_ratio = est_lds / actual.lds_bytes
            assert lds_ratio <= 1.15, (
                f"LDS estimate {est_lds} is {lds_ratio:.2f}x actual {actual.lds_bytes} -- >15% over"
            )

        # VGPRs: within [0.65, 1.5] of actual (structural model, regalloc varies)
        if actual.vgpr_count > 0:
            vgpr_ratio = est_vgprs / actual.vgpr_count
            assert 0.65 <= vgpr_ratio <= 1.5, (
                f"VGPR estimate {est_vgprs} vs actual {actual.vgpr_count} (ratio {vgpr_ratio:.2f})"
            )

        # AGPRs: exact match (purely tile-shape determined)
        assert est_agprs == actual.agpr_count, f"AGPR estimate {est_agprs} != actual {actual.agpr_count}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--wg", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--wpw", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--twg", type=int, nargs=3, default=[3, 2, 1])
    parser.add_argument("--k-mult", type=int, default=4, help="K = k_mult * k_t * transfer_tile_k_elems")
    parser.add_argument("--pipeline-strategy", type=int, default=1)
    args = parser.parse_args()

    cfg = _make_instance(args.wg, args.wpw, args.twg, args.k_mult, args.pipeline_strategy)
    gs = cfg.gemm_size
    tag = f"wg{'x'.join(map(str, args.wg))}_w{'x'.join(map(str, args.wpw))}_t{'x'.join(map(str, args.twg))}"
    print(f"Config: {tag}_km{args.k_mult}_ps{args.pipeline_strategy}")
    print(f"  M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        _, asm = compile_multitile_gemm(
            cfg,
            f.name,
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        )
    if args.print_asm:
        print(asm)


class TestILPScheduler:
    """Real ILP-scheduler coverage (ll_ilp_sched=2) on a small config set, so the
    ILP path is exercised without doubling the greedy correctness sweeps above."""

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [([2, 2, 1], [6, 4, 1]), ([4, 2, 1], [12, 6, 1])],
        ids=["4w_6x4", "8w_12x6"],
    )
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg):
        cfg = _make_instance(
            [1, 1, 1],
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult=4,
            pipeline_strategy=3,
            ll_ilp_sched=2,
        )
        _run_multitile(cfg)
