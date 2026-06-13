"""CDNA4 (MI350, gfx950) sanity test on the non-G2S code path, direct_b operand_path only, mcpu=gfx950."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np
import pytest

from aster import ir
import tempfile

from aster.layout import Layout, LayoutValues, Swizzle, Symbol, Tensor, enumerate_flat_coords, tile
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


# Compute / WG axes (layout-driven kernel body, mirrors directb_cdna3).
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# Cooperative A-load inner-iter axes.
m_load_a = Symbol("m_load_a")
k_load_a = Symbol("k_load_a")
# Direct preshuffled B per-wave per-iter tile axes (each wave loads its own
# n_per_wave x k_t tiles, no cooperation).
n_load_b = Symbol("n_load_b")
k_load_b = Symbol("k_load_b")


def _build_multitile_gemm(cfg: "MultitileGemmInstance") -> ir.Module:
    """Build a multi-tile multi-wave multi-WG pipelined GEMM kernel (CDNA4 direct-B).

    Specialized for direct_b: B bypasses LDS (per-wave global load at
    preshuffle byte offsets, vx4 split into 2 vx2 fragments for MFMA).
    """
    from kittens_helpers import PIPELINE_STRATEGIES
    from aster.dialects.kernel_builder import MFMA_F16_CDNA4
    from aster._mlir_libs._amdgcn import AGPRRangeType

    spec, mapping = cfg.spec, cfg.mapping

    # --- Sizes + tile geometry from spec/mapping ---
    M_total = spec.gemm_size[DIM_M]
    N_total = spec.gemm_size[DIM_N]
    K = spec.gemm_size[DIM_K]
    mfma_m, mfma_n, mfma_k = (
        spec.mfma_shape[DIM_M],
        spec.mfma_shape[DIM_N],
        spec.mfma_shape[DIM_K],
    )
    assert mfma_k == MFMA_F16_CDNA4.shape[2], f"cdna4 leaf requires mfma_k={MFMA_F16_CDNA4.shape[2]}, got {mfma_k}"
    elt_bytes_a = spec.elt_bytes_a

    wg = mapping.num_workgroups_per_kernel
    wpw = mapping.num_waves_per_workgroup
    twg = mapping.num_tiles_per_workgroup
    wg_m_count, wg_n_count = wg[DIM_M], wg[DIM_N]
    wpw_m, wpw_n = wpw[DIM_M], wpw[DIM_N]
    m_t, n_t, k_t = twg[DIM_M], twg[DIM_N], twg[DIM_K]
    nw = wpw_m * wpw_n
    ws = mapping.wave_size

    assert m_t % wpw_m == 0, f"twg_m={m_t} not divisible by wpw_m={wpw_m}"
    assert n_t % wpw_n == 0, f"twg_n={n_t} not divisible by wpw_n={wpw_n}"
    m_per_wave = m_t // wpw_m
    n_per_wave = n_t // wpw_n

    # Pipeline stages -- B_LDS_WRITE / B_LDS_READ entries are ignored
    # (B is direct, no LDS path).
    stg = PIPELINE_STRATEGIES[mapping.pipeline_strategy]
    STG_A_LOAD = stg["A_LOAD"]
    STG_A_WRITE = stg["A_LDS_WRITE"]
    STG_A_READ = stg["A_LDS_READ"]
    STG_B_LOAD = stg["B_LOAD"]
    STG_COMPUTE = stg["COMPUTE"]
    N_STAGES = STG_COMPUTE + 1

    tile_k_elems = cfg.transfer_tile_k_elems  # = (ws/mfma_m) * (glb_bytes/elt_bytes_a)
    tile_bytes_a = cfg.transfer_tile_bytes  # = mfma_m * tile_k_elems * elt_bytes_a
    frag_k = mfma_k // 2  # K elements per ds_read_b64 vx2 fragment
    n_frags = tile_k_elems // frag_k  # vx2 fragments per K-tile (= 2)

    k_step = k_t * tile_k_elems
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, (
        f"k_iters={k_iters} < N_STAGES={N_STAGES}; pipeliner needs at least N_STAGES iters for prologue+steady-state"
    )

    # Operand strides (leading-dim byte stride).
    ol_a = spec.operand_layout(OP_A)
    ol_c = spec.operand_layout(OP_C)
    stride_a = ol_a.strides[0]
    stride_c_row, stride_c_col = ol_c.strides[0], ol_c.strides[1]

    # Direct-B preshuffle constants. Each (n_tile, k_tile) is ws lanes x
    # global_load_bytes per lane. Matches kittens_helpers.shuffle_weight.
    lane_s = mapping.global_load_bytes
    k_tile_bytes_pres = ws * lane_s
    assert K % tile_k_elems == 0
    n_tile_bytes_pres = (K // tile_k_elems) * k_tile_bytes_pres

    # Shared WG-level LDS for A (only -- B is direct).
    lds_total_a = m_t * k_t * tile_bytes_a

    # CDNA4 LDS swizzle. shift=4 matches mapping.ds_write_bytes=8 stride.
    LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=4)

    b = KernelBuilder("gemm_mod", cfg.kernel_name, target=mapping.mcpu)
    b.set_block_dims(mapping.num_threads)
    b.set_grid_dims(mapping.num_workgroups)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # --- TiledCopies ---
    ds_write_bytes = mapping.ds_write_bytes
    ds_read_bytes = mapping.ds_read_bytes
    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout((mfma_m, ws // mfma_m), (stride_a, lane_s)),
        value_layout=Layout(1, 0),
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_write_64b,
        thread_layout=Layout((mfma_m, ws // mfma_m), (ws, lane_s)),
        value_layout=Layout(lane_s // ds_write_bytes, ds_write_bytes),
        swizzle=LDS_SWIZZLE,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((ws // mfma_m, mfma_m), (ds_read_bytes, ws)),
        value_layout=Layout(n_frags, frag_k * elt_bytes_a),
        swizzle=LDS_SWIZZLE,
    )
    # Direct B: 64 lanes each load 16 bytes (dwordx4 = full vx4 MFMA operand)
    # contiguous within the preshuffled tile -- no LDS, no swizzle.
    tc_load_b_direct = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout(ws, lane_s),
        value_layout=Layout(1, 0),
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

    # --- Layouts ---
    # Global A: hierarchical (mfma_m, m_t) x (tile_k_elems, k_t).
    A_TILED = tile(
        Layout((M_total, K), (stride_a, elt_bytes_a)),
        tile_sizes=((mfma_m, m_t), (tile_k_elems, k_t)),
        axes=((m, wg_m), (k_tile, global_k)),
    )
    # Global preshuffle B: per-tile strides in the shuffled byte order.
    B_TILED = Layout(
        sizes=(n_per_wave, wpw_n, wg_n_count, k_t, k_iters),
        strides=(
            n_tile_bytes_pres,
            n_per_wave * n_tile_bytes_pres,
            n_t * n_tile_bytes_pres,
            k_tile_bytes_pres,
            k_t * k_tile_bytes_pres,
        ),
        axes=(n_load_b, wave_n, wg_n, k_load_b, global_k),
    )
    C_TILED = tile(
        Layout((M_total, N_total), (stride_c_row, stride_c_col)),
        tile_sizes=((mfma_m, m_per_wave, wpw_m), (mfma_n, n_per_wave, wpw_n)),
        axes=((m, wave_m, wg_m), (n, wave_n, wg_n)),
    )

    # Shared WG-level LDS read view for A.
    LDS_A_READ_TILED = tile(
        Layout((m_t, k_t * tile_bytes_a), (k_t * tile_bytes_a, 1)),
        tile_sizes=((1, m_per_wave), tile_bytes_a),
        axes=((m, wave_m), k_tile),
    )

    # Per-WG and per-wave distributions; cooperative-load plan for A.
    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg_m_count, wg_n_count))
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    plan_a = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((m_t, k_t), (mfma_m * stride_a, tile_k_elems * elt_bytes_a)),
        wg_tile_lds=Layout((m_t, k_t), (k_t * tile_bytes_a, tile_bytes_a)),
        spatial_axis=m_load_a,
        k_axis=k_load_a,
    )

    # Pre-sliced tensors (same slice form as test_101d/e/f).
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {wg_n: wg_n_idx, wave_n: wave_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {wg_m: wg_m_idx, wg_n: wg_n_idx, wave_m: wave_m_idx, wave_n: wave_n_idx},
    )

    n_accs = m_per_wave * n_per_wave

    # Accumulators live in a memref (c_buf) -- one entry per (mi, ni) acc,
    # loaded/stored per MFMA. Keeps only ONE acc-vx4 live at a time instead of
    # all n_accs through scf.for iter_args, which blows the AGPR budget for deep
    # pipelines + big tiles.
    ax4_type = AGPRRangeType.get(b._ctx, size=4)
    c_buf = b.memref_alloca(b.constant_index(n_accs), ax4_type)
    for i in range(n_accs):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, b.constant_index(i))

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)

    @b.loop(c0, b.constant_index(k_iters), c1)
    def body(k_iv):
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
            assert n_frags == 2, f"cdna4 expects 2 vx2 frags per K-tile, got {n_frags}"
            for ki, mi, ni in enumerate_flat_coords((k_t, m_per_wave, n_per_wave)):
                acc_idx = b.constant_index(mi * n_per_wave + ni)
                acc = b.memref_load(c_buf, acc_idx)
                a_d = b.join_vx2_to_vx4(
                    a_frags.data_at((mi, ki, 0)),
                    a_frags.data_at((mi, ki, 1)),
                )
                b_d = b_vx4.data_at((ni, ki))
                new_acc = b.mfma(MFMA_F16_CDNA4.opcode, acc, a_d, b_d)
                b.memref_store(new_acc, c_buf, acc_idx)
            b.dealloc_lds(lds_a_h)

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

    Preshuffles B (direct_b operand path). Returns (C_output, times_ns).
    """
    from kittens_helpers import shuffle_weight

    mcpu = cfg.mapping.mcpu
    if not skip_gpu_check and not system_has_gpu(mcpu):
        pytest.skip(f"GPU {mcpu} not available, skip execution")

    B_gpu = shuffle_weight(B)
    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)
    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        arguments=[InputArray(A.flatten()), InputArray(B_gpu.flatten()), OutputArray(C_output)],
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
        operand_path=OperandPath.DIRECT_B,
        rotate_compute_stage=rotate_compute_stage,
        ll_sched=ll_sched,
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


class TestCdna4Mfma:
    """CDNA4 v_mfma_f32_16x16x32_f16 via the non-G2S memory path (mcpu=gfx950).

    Same global_load -> ds_write -> ds_read path as CDNA3, but the
    compute loop joins 2 vx2 fragments into vx4 for the doubled-K MFMA.
    """

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([1, 1, 1], [2, 2, 1]),
            ([2, 2, 1], [4, 4, 1]),
        ],
        ids=["1w_2x2", "4w_4x4"],
    )
    @pytest.mark.parametrize("pipeline_strategy", [0, 3], ids=["ps0", "ps3"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg, pipeline_strategy, rotate_compute_stage):
        cfg = _make_instance(
            [1, 1, 1],
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult=4,
            pipeline_strategy=pipeline_strategy,
            mcpu="gfx950",
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_multitile(cfg)

    def test_operand_path(self):
        cfg = _make_instance(
            [1, 1, 1],
            [2, 2, 1],
            [4, 4, 1],
            k_mult=4,
            pipeline_strategy=3,
            mcpu="gfx950",
        )
        _run_multitile(cfg)

    def test_multi_wg(self):
        cfg = _make_instance(
            [2, 2, 1],
            [2, 2, 1],
            [4, 4, 1],
            k_mult=4,
            pipeline_strategy=3,
            mcpu="gfx950",
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
            mcpu="gfx950",
            ll_sched=ll_sched,
        )
        _run_multitile(cfg)
