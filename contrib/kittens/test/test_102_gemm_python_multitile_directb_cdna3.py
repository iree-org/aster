import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile

import numpy as np
import pytest

from aster import ir
from aster.layout import Layout, Swizzle, Symbol, Tensor, enumerate_flat_coords, tile
from coop import CoopLoadPlan, coop_2d_split, make_coop_load_plan  # noqa: F401
from aster.dialects.kernel_builder_with_layouts import (
    ds_read_64b,
    ds_write_64b,
    global_load_dwordx4,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_gpu
from aster.pass_pipelines import make_default_pass_pipeline

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


# Compute / WG axes (layout-driven kernel body, mirrors test_101f).
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
    """Build a multi-tile multi-wave multi-WG pipelined GEMM kernel.

    Direct-B variant: B bypasses LDS (per-wave global load at preshuffle
    byte offsets, vx4 split into 2 vx2 fragments for MFMA). Layout-driven
    kernel body (matches test_101f's style): Symbol-typed axes,
    `Tensor`/`slice()`/`transfer_tiles`, and `CoopLoadPlan` for A's
    cooperative load. No `define_helper` / `foreach_tile` / `to_any`.

    The kernel honors `cfg.mapping.pipeline_strategy` via
    `PIPELINE_STRATEGIES`; B's LDS stages are unused (B is direct).
    """
    from kittens_helpers import PIPELINE_STRATEGIES

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
    assert (mfma_m, mfma_n, mfma_k) == (16, 16, 16), (
        f"Only MFMA 16x16x16 supported in this kernel (got {(mfma_m, mfma_n, mfma_k)})"
    )
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

    # CDNA3 LDS swizzle. shift=4 matches mapping.ds_write_bytes=8 stride.
    lds_swizzle = Swizzle(bits=3, base=3, shift=4)

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
        swizzle=lds_swizzle,
    )
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((ws // mfma_m, mfma_m), (ds_read_bytes, ws)),
        value_layout=Layout(tile_k_elems // mfma_k, mfma_k * elt_bytes_a),
        swizzle=lds_swizzle,
    )
    # Direct B: 64 lanes each load 16 bytes contiguous within the
    # preshuffled tile -- no LDS, no swizzle.
    tc_load_b_direct = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout(ws, lane_s),
        value_layout=Layout(1, 0),
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

    # Global C-store layout (n_agprs derived from ws/mfma_n).
    n_agprs = ws // mfma_n
    GLOBAL_STORE_TILE_C = Layout(
        (n_agprs, mfma_n, n_agprs),
        (n_agprs * stride_c_row, stride_c_col, stride_c_row),
    )
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    n_accs = m_per_wave * n_per_wave
    n_frags = tile_k_elems // mfma_k

    # Accumulators live in a memref (c_buf) -- one entry per (mi, ni) acc,
    # loaded/stored per MFMA. Keeps only ONE acc-vx4 live at a time
    # instead of all n_accs through scf.for iter_args, which blows the
    # AGPR budget for deep pipelines + big tiles (m_per_wave * n_per_wave
    # up to 32 in TestPipelineDirectBLLSched).
    from aster._mlir_libs._amdgcn import AGPRRangeType

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
            bfence = b.cross_wave_token_barrier(a_write)
            sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
            a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile), fence_token=bfence)

        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_vx4)
            # MFMA loop: split B vx4 per use (compiler CSEs the split).
            # Each acc loaded from c_buf, updated, stored back -- only one
            # acc-vx4 SSA value live across the MFMA at a time.
            for fi, ki, mi, ni in enumerate_flat_coords((n_frags, k_t, m_per_wave, n_per_wave)):
                ai = mi * n_per_wave + ni
                acc_idx = b.constant_index(ai)
                a_d = a_frags.data_at((mi, ki, fi))
                b_lo, b_hi = b.split_register_range(b_vx4.data_at((ni, ki)), 2)
                b_d = b_lo if fi == 0 else b_hi
                acc = b.memref_load(c_buf, acc_idx)
                new_acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_d, b_d)
                b.memref_store(new_acc, c_buf, acc_idx)
            b.dealloc_lds(lds_a_h)

    for mi, ni in enumerate_flat_coords((m_per_wave, n_per_wave)):
        ai = mi * n_per_wave + ni
        acc = b.memref_load(c_buf, b.constant_index(ai))
        b.store_multi_fragment_to_global(
            acc,
            c_ptr,
            b.slice(TC, {m: mi, n: ni}).offset,
            GLOBAL_STORE_TILE_C,
            GLOBAL_STORE_SUB_TILE_C,
            b.global_store_dword,
        )
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

    @property
    def preshuffle_n_blocks(self) -> int:
        """N / mfma_n: logical blocks along the N dimension for preshuffled B."""
        return self.gemm_size[DIM_N] // self.spec.mfma_shape[DIM_N]

    @property
    def preshuffle_k_blocks(self) -> int:
        """K / transfer_tile_k_elems: K-tile count matching shuffle_weight chunking."""
        return self.gemm_size[DIM_K] // self.transfer_tile_k_elems

    @property
    def preshuffle_lane_stride_bytes(self) -> int:
        """Byte stride between consecutive lane_ids in preshuffled B (global dwordx4 width)."""
        return self.mapping.global_load_bytes

    @property
    def preshuffle_k_block_stride_bytes(self) -> int:
        """Byte stride between K-tile blocks: one full wave plane of dwordx4 loads."""
        return self.mapping.wave_size * self.preshuffle_lane_stride_bytes


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

    Automatically preshuffles B when cfg.mapping.direct_b is True.
    Returns (C_output, times_ns).
    """
    from kittens_helpers import shuffle_weight

    mcpu = getattr(cfg.mapping, "mcpu", "gfx942")
    if not skip_gpu_check and not system_has_gpu(mcpu):
        pytest.skip(f"GPU {mcpu} not available, skip execution")

    B_gpu = shuffle_weight(B) if cfg.mapping.direct_b else B
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
    """Operand path (direct_b) sweep, orthogonal to geometry/pipeline."""

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


class TestPipelineDirectBLLSched:
    """Ps sweep for b_path=direct_b with ll_sched=1, multi-WG geometry.

    Note: ps5/6/7 + (1,4)-wave + (8,16,1)-tile + k_mult=128 currently fails
    register allocation. The pre-refactor (define_helper + foreach_tile +
    any_type buffering) gave SROA enough room to prune buffered-fragment
    lifetimes; the layout-driven Result approach holds the buffered vx4 /
    vx2 entries as register-typed memrefs that the regalloc can't prune
    the same way. Restoring `any_type` wrapping is a separate refactor.
    """

    @pytest.mark.parametrize(
        "pipeline_strategy",
        [
            1,
            3,
            4,
            pytest.param(5, marks=pytest.mark.xfail(reason="regalloc: deep pipeline + 32 accs + Result-typed buffers")),
            pytest.param(6, marks=pytest.mark.xfail(reason="regalloc: deep pipeline + 32 accs + Result-typed buffers")),
            pytest.param(7, marks=pytest.mark.xfail(reason="regalloc: deep pipeline + 32 accs + Result-typed buffers")),
        ],
        ids=lambda p: f"ps{p}",
    )
    def test_correctness(self, pipeline_strategy):
        cfg = _make_instance(
            [19, 16, 1],
            [1, 4, 1],
            [8, 16, 1],
            k_mult=128,
            pipeline_strategy=pipeline_strategy,
            ll_sched=1,
        )
        _run_multitile(cfg)


class TestLLSched:
    """LL-scheduler preset axis (ll_sched 0..5) on a register-safe 4w geometry.

    0 = scheduler off; 1 = default mfma-hiding preset; 3/5 engage the
    deterministic xdlMaxRun interleaving cap (3 = mid, 5 =
    max-1-contiguous). Geometry mirrors the proven 4w_6x4 ps3 config
    used in TestPipeline, so all presets allocate cleanly (no deep-
    pipeline regalloc pressure).
    """

    @pytest.mark.parametrize("ll_sched", [0, 1, 3, 5], ids=lambda v: f"llsched{v}")
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
        ([1, 1, 1], [1, 1, 1], [3, 2, 1], 4, 1, "1w_ps1_directb"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 1, "4w_ps1_directb"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 3, "4w_ps3_directb"),
        ([1, 1, 1], [2, 2, 1], [4, 4, 1], 4, 4, "4w_ps4_directb"),
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
        - VGPRs: within factor 2 (estimate is structural, regalloc varies).
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

        # VGPRs: within factor 2 (regalloc can vary)
        if actual.vgpr_count > 0:
            vgpr_ratio = est_vgprs / actual.vgpr_count
            assert 0.5 <= vgpr_ratio <= 2.0, (
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
