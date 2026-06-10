"""CDNA4 (MI350, gfx950) multi-tile GEMM using KernelBuilderWithLayouts - direct_b variant.

Specialization of test_102_gemm_python_multitile_cdna4 that keeps only the
direct_b operand path: A goes through LDS (G2S), B goes through registers via
a preshuffled global load (no LDS for B).

Same structure as test_102_gemm_python_multitile but targeting CDNA4:
  - v_mfma_f32_16x16x32_f16 (doubled-K, 4 VGPR A/B operands)
  - ds_read_b64 (64-bit LDS reads for vx2 MFMA fragments)
  - 2 fragment per tile

Memory path A: G2S buffer_load_dwordx4_lds -> ds_read_b64 (2x, joined to vx4) -> MFMA.
Memory path B: preshuffled buffer_load_dwordx4 -> split vx4 into 2 vx2 frags -> MFMA.
"""

import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder import MFMA_F16_CDNA4
from aster.dialects.kernel_builder_with_layouts import (
    ds_read_64b,
    global_load_dwordx4,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_gpu
from aster.pass_pipelines import make_default_pass_pipeline
from aster.layout import Layout, Swizzle, Symbol, Tensor, enumerate_flat_coords, tile
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
    LoadType,
    OperandPath,
    WeakScaledMappedGemmInstance,
)


# Compute / WG axes (layout-driven kernel body, mirrors test_102 cdna3 / test_101f).
m = Symbol("m")
n = Symbol("n")
k_tile = Symbol("k_tile")
global_k = Symbol("global_k")
wg_m = Symbol("wg_m")
wg_n = Symbol("wg_n")
wave_m = Symbol("wave_m")
wave_n = Symbol("wave_n")
# Cooperative A G2S inner-iter axes.
m_load_a = Symbol("m_load_a")
k_load_a = Symbol("k_load_a")
# Direct preshuffled B per-wave per-iter tile axes.
n_load_b = Symbol("n_load_b")
k_load_b = Symbol("k_load_b")


def _build_cdna4_gemm(cfg: "Cdna4GemmInstance") -> ir.Module:
    """Build a CDNA4 multi-tile GEMM kernel: G2S A + direct preshuffled B.

    G2S collapses LOAD + LDS_WRITE into one stage. A_LDS_WRITE, B_LDS_WRITE and
    B_LDS_READ parts of the pipeline strategy are unused.
    """
    from kittens_helpers import PIPELINE_STRATEGIES

    spec, mapping = cfg.spec, cfg.mapping
    assert mapping.direct_b, "this variant only supports direct_b path"

    # --- Sizes + tile geometry from spec/mapping ---
    M_total = spec.gemm_size[DIM_M]
    N_total = spec.gemm_size[DIM_N]
    K = spec.gemm_size[DIM_K]
    mfma_m, mfma_n, mfma_k = (
        spec.mfma_shape[DIM_M],
        spec.mfma_shape[DIM_N],
        spec.mfma_shape[DIM_K],
    )
    assert (mfma_m, mfma_n) == (16, 16) and mfma_k == 32, (
        f"CDNA4 g2s_directb kernel only supports MFMA 16x16x32 (got {(mfma_m, mfma_n, mfma_k)})"
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

    # Pipeline stages -- A_LDS_WRITE / B_LDS_WRITE / B_LDS_READ entries
    # ignored (G2S collapses A LOAD+WRITE, B is direct).
    stg = PIPELINE_STRATEGIES[mapping.pipeline_strategy]
    STG_A_LOAD = stg["A_LOAD"]
    STG_A_LDS_READ = stg["A_LDS_READ"]
    STG_B_LOAD = stg["B_LOAD"]
    STG_COMPUTE = stg["COMPUTE"]
    N_STAGES = STG_COMPUTE + 1
    first_lds_load = min(STG_A_LOAD, STG_B_LOAD)

    tile_k_elems = cfg.transfer_tile_k_elems
    tile_row_bytes = cfg.transfer_tile_row_bytes
    tile_bytes_a = cfg.transfer_tile_bytes
    frag_k = mfma_k // 2  # K elements per ds_read fragment (= 16, half-K).
    n_frags_per_tile = tile_k_elems // frag_k  # = 2

    k_step = k_t * tile_k_elems
    assert K % k_step == 0, f"K={K} not divisible by k_step={k_step}"
    k_iters = K // k_step
    assert k_iters >= N_STAGES, f"k_iters={k_iters} < N_STAGES={N_STAGES}; pipeliner needs at least N_STAGES iters"

    # Operand strides.
    ol_a = spec.operand_layout(OP_A)
    ol_c = spec.operand_layout(OP_C)
    stride_a = ol_a.strides[0]
    stride_c_row, stride_c_col = ol_c.strides[0], ol_c.strides[1]

    # Direct-B preshuffle constants.
    lane_s = mapping.global_load_bytes
    k_tile_bytes_pres = ws * lane_s
    n_tile_bytes_pres = (K // tile_k_elems) * k_tile_bytes_pres

    # Shared per-WG LDS for A.
    lds_total_a = m_t * k_t * tile_bytes_a
    lds_swizzle = Swizzle(bits=2, base=4, shift=4)

    b = KernelBuilder("gemm_cdna4_mod", cfg.kernel_name, target=mapping.mcpu)
    b.set_block_dims(mapping.num_threads)
    b.set_grid_dims(mapping.num_workgroups)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # --- TiledCopies (used for LDS read A + direct B FLAT load) ---
    ds_read_bytes = mapping.ds_read_bytes
    tc_dsr_a = b.make_tiled_copy_descriptor(
        ds_read_64b,
        thread_layout=Layout((ws // mfma_m, mfma_m), (ds_read_bytes, tile_row_bytes)),
        value_layout=Layout(n_frags_per_tile, frag_k * elt_bytes_a),
        swizzle=lds_swizzle,
    )
    tc_load_b_direct = b.make_tiled_copy_descriptor(
        global_load_dwordx4,
        thread_layout=Layout(ws, lane_s),
        value_layout=Layout(1, 0),
    )

    # --- Layouts ---
    A_TILED = tile(
        Layout((M_total, K), (stride_a, elt_bytes_a)),
        tile_sizes=((mfma_m, m_t), (tile_k_elems, k_t)),
        axes=((m, wg_m), (k_tile, global_k)),
    )
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
    LDS_A_READ_TILED = tile(
        Layout((m_t, k_t * tile_bytes_a), (k_t * tile_bytes_a, 1)),
        tile_sizes=((1, m_per_wave), tile_bytes_a),
        axes=((m, wave_m), k_tile),
    )

    # --- Distribution + cooperative-load plan for A ---
    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg_m_count, wg_n_count))
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw_m, wpw_n))

    plan_a = make_coop_load_plan(
        b,
        wid,
        num_waves=nw,
        wg_tile_global=Layout((m_t, k_t), (mfma_m * stride_a, tile_row_bytes)),
        wg_tile_lds=Layout((m_t, k_t), (k_t * tile_bytes_a, tile_bytes_a)),
        spatial_axis=m_load_a,
        k_axis=k_load_a,
    )

    # Per-WG sliced tensors.
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {wg_m: wg_m_idx})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {wg_n: wg_n_idx, wave_n: wave_n_idx})
    TC = b.slice(
        Tensor(c_ptr, layout=C_TILED),
        {wg_m: wg_m_idx, wg_n: wg_n_idx, wave_m: wave_m_idx, wave_n: wave_n_idx},
    )

    # Global C-store tile layout.
    n_agprs = ws // mfma_n
    GLOBAL_STORE_TILE_C = Layout(
        (n_agprs, mfma_n, n_agprs),
        (n_agprs * stride_c_row, stride_c_col, stride_c_row),
    )
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    # --- G2S setup (buffer resource + M0 + soff + per-thread tile-local off),
    # bundled into a TransferTileG2S by prepare_transfer_tiles_g2s.
    g2s_a = b.prepare_transfer_tiles_g2s(
        a_ptr,
        buffer_num_records_bytes=M_total * stride_a,
        spatial_dim=mfma_m,
        tile_row_bytes=tile_row_bytes,
        global_load_bytes=mapping.global_load_bytes,
        global_row_stride=stride_a,
        swizzle=lds_swizzle,
    )

    # --- Accumulators are explicitly set in c_buf to avoid iter_args register
    # pressure that blows up regalloc for deep pipelines.
    from aster._mlir_libs._amdgcn import AGPRRangeType

    n_accs = m_per_wave * n_per_wave
    ax4_type = AGPRRangeType.get(b._ctx, size=4)
    c_buf = b.memref_alloca(b.constant_index(n_accs), ax4_type)
    for i in range(n_accs):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, b.constant_index(i))

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)

    @b.loop(c0, b.constant_index(k_iters), c1)
    def body(k_iv):
        # -- LDS alloc (only A; B is direct_b).
        with b.stage(first_lds_load):
            lds_a_h, sA_full = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_READ_TILED)

        # -- G2S LOAD A: cooperative, per-tile loop via transfer_tiles_g2s.
        with b.stage(STG_A_LOAD):
            ta_iter = b.slice(TA, {global_k: k_iv})
            g2s_toks_a = b.transfer_tiles_g2s(
                Tensor(
                    a_ptr,
                    b.layout_sum(ta_iter.offset, plan_a.global_wave_off),
                    plan_a.global_layout,
                ),
                Tensor(sA_full.ptr, plan_a.lds_wave_off, plan_a.lds_layout),
                g2s_a,
                unroll_axes=plan_a.unroll_axes,
            )

        # -- LOAD B (direct preshuffled FLAT load via TiledCopy).
        with b.stage(STG_B_LOAD):
            tb_iter = b.slice(TB, {global_k: k_iv})
            b_vx4_loads = b.transfer_tiles(tb_iter, tc_load_b_direct, unroll_axes=(n_load_b, k_load_b))

        # -- LDS READ A: wait G2S + barrier + read via tc_dsr_a + dealloc.
        with b.stage(STG_A_LDS_READ):
            b.wait_deps(*g2s_toks_a)
            b.s_barrier()
            sA_read = b.slice(sA_full, {wave_m: wave_m_idx})
            a_frags = b.transfer_tiles(sA_read, tc_dsr_a, unroll_axes=(m, k_tile))
            b.dealloc_lds(lds_a_h)

        # -- COMPUTE: join 2 vx2 A frags into vx4, MFMA with vx4 B directly.
        with b.stage(STG_COMPUTE):
            b.wait_deps(a_frags, b_vx4_loads)
            for kt, mi, ni in enumerate_flat_coords((k_t, m_per_wave, n_per_wave)):
                acc_idx = b.constant_index(mi * n_per_wave + ni)
                a_lo = a_frags.data_at((mi, kt, 0))
                a_hi = a_frags.data_at((mi, kt, 1))
                a_vx4 = b.join_vx2_to_vx4(a_lo, a_hi)
                b_vx4 = b_vx4_loads.data_at((ni, kt))
                acc = b.memref_load(c_buf, acc_idx)
                new_acc = b.mfma(MFMA_F16_CDNA4.opcode, acc, a_vx4, b_vx4)
                b.memref_store(new_acc, c_buf, acc_idx)

            # WAR sync: without this, a fast wave's next-iter G2S can
            # overwrite LDS offsets that a slow wave is still ds_read'ing.
            b.s_barrier()

    # Store C tiles.
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


KERNEL_NAME = "gemm_cdna4"


class Cdna4GemmInstance(WeakScaledMappedGemmInstance):
    """Config with CDNA4 MFMA shape and transfer-tile geometry."""

    @property
    def kernel_name(self) -> str:
        return KERNEL_NAME

    @property
    def transfer_tile_k_elems(self) -> int:
        return self.spec.mfma_shape[DIM_K]

    @property
    def transfer_tile_row_bytes(self) -> int:
        return self.transfer_tile_k_elems * self.spec.elt_bytes_a

    @property
    def transfer_tile_bytes(self) -> int:
        return self.spec.mfma_shape[DIM_M] * self.transfer_tile_row_bytes

    @property
    def preshuffle_n_blocks(self) -> int:
        n, mn = self.gemm_size[DIM_N], self.spec.mfma_shape[DIM_N]
        assert n % mn == 0, f"N={n} not divisible by mfma_n={mn}"
        return n // mn

    @property
    def preshuffle_k_blocks(self) -> int:
        k, tk = self.gemm_size[DIM_K], self.transfer_tile_k_elems
        assert k % tk == 0, f"K={k} not divisible by transfer_tile_k_elems={tk}"
        return k // tk

    @property
    def preshuffle_lane_stride_bytes(self) -> int:
        return self.mapping.global_load_bytes

    @property
    def preshuffle_k_block_stride_bytes(self) -> int:
        return self.mapping.wave_size * self.preshuffle_lane_stride_bytes

    @property
    def label(self) -> str:
        return f"{super().label}_cdna4"

    @classmethod
    def from_label(cls, label: str) -> "Cdna4GemmInstance":
        if label.endswith("_cdna4"):
            label = label[: -len("_cdna4")]
        base = WeakScaledMappedGemmInstance.from_label(label)
        spec = GemmSpec.from_sizes(*base.gemm_size, mfma_shape=list(MFMA_F16_CDNA4.shape))
        mapping = dataclasses.replace(
            base.mapping,
            operand_path=OperandPath.DIRECT_B,
            load_type=LoadType.FLAT,
        )
        return cls(spec, mapping)


def _make_instance(
    num_workgroups,
    num_waves_per_wg,
    num_tiles_per_wg,
    k_mult,
    pipeline_strategy=0,
    rotate_compute_stage=False,
):
    """Build a Cdna4GemmInstance from tile grid and K multiplier."""
    mfma = MFMA_F16_CDNA4.shape
    twg_m, twg_n = num_tiles_per_wg[DIM_M], num_tiles_per_wg[DIM_N]
    wpw_m, wpw_n = num_waves_per_wg[DIM_M], num_waves_per_wg[DIM_N]
    assert twg_m % wpw_m == 0 and twg_n % wpw_n == 0
    M = num_workgroups[DIM_M] * twg_m * mfma[DIM_M]
    N = num_workgroups[DIM_N] * twg_n * mfma[DIM_N]
    K = k_mult * mfma[DIM_K]
    spec = GemmSpec.from_sizes(M, N, K, mfma_shape=list(mfma))
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[twg_m // wpw_m, twg_n // wpw_n, num_tiles_per_wg[DIM_K]],
        pipeline_strategy=pipeline_strategy,
        operand_path=OperandPath.DIRECT_B,
        rotate_compute_stage=rotate_compute_stage,
        mcpu="gfx950",
    )
    return Cdna4GemmInstance(spec, mapping)


def compile_cdna4_gemm(cfg, output_hsaco_path, **kw):
    """Compile a CDNA4 GEMM config to HSACO."""
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_cdna4_gemm(cfg)
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


def execute_cdna4_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled CDNA4 GEMM HSACO.

    Automatically preshuffles B when ``cfg.mapping.direct_b`` is True.
    """
    from kittens_helpers import shuffle_weight

    mcpu = getattr(cfg.mapping, "mcpu", "gfx950")
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


def _run_cdna4_gemm(cfg):
    """Compile + run a CDNA4 GEMM, verify against numpy."""
    gs = cfg.gemm_size

    np.random.seed(42 + gs[DIM_M] + gs[DIM_N] + gs[DIM_K])
    A_mat = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
    B_mat = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
        compile_cdna4_gemm(cfg, tmp.name)
        C_output, _ = execute_cdna4_hsaco(cfg, tmp.name, 1, A_mat, B_mat)

    expected = (A_mat.astype(np.float32) @ B_mat.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


def _min_k_iters_for_ps(k_t: int, ps: int) -> int:
    """Minimum K iterations for a pipeline strategy (stages+1 warmup cycles)."""
    from kittens_helpers import PIPELINE_STRATEGIES

    return max(PIPELINE_STRATEGIES[ps].values()) + 1


class TestCdna4GemmG2S:
    """Known-good configs: wpw=twg, tpw=[1,1,1] (matches old test_102_cdna4)."""

    @pytest.mark.parametrize(
        "twg",
        [[2, 2, 1], [2, 1, 1], [1, 2, 1]],
        ids=["2x2", "2x1", "1x2"],
    )
    @pytest.mark.parametrize("k_mult", [2, 4, 8], ids=["km2", "km4", "km8"])
    def test_correctness(self, twg, k_mult):
        cfg = _make_instance([1, 1, 1], twg, twg, k_mult)
        _run_cdna4_gemm(cfg)

    @pytest.mark.parametrize(
        "twg",
        [[2, 2, 1], [2, 1, 1], [1, 2, 1]],
        ids=["2x2", "2x1", "1x2"],
    )
    def test_correctness_km1(self, twg):
        cfg = _make_instance([1, 1, 1], twg, twg, 1)
        _run_cdna4_gemm(cfg)


class TestCdna4MultiWG:
    """Multi-workgroup correctness."""

    @pytest.mark.parametrize(
        "num_workgroups,twg",
        [
            ([2, 2, 1], [2, 2, 1]),
        ],
        ids=["mwg2x2_4w_2x2"],
    )
    def test_correctness(self, num_workgroups, twg):
        cfg = _make_instance(num_workgroups, twg, twg, k_mult=4)
        _run_cdna4_gemm(cfg)


class TestCdna4Coop8Waves:
    """8-wave cooperative loading correctness."""

    @pytest.mark.parametrize(
        "wpw,twg",
        [
            ([2, 4, 1], [4, 16, 2]),
            ([4, 2, 1], [16, 4, 2]),
            ([1, 8, 1], [2, 32, 2]),
            ([8, 1, 1], [32, 2, 2]),
        ],
        ids=["w2x4_twg4x16x2", "w4x2_twg16x4x2", "w1x8_twg2x32x2", "w8x1_twg32x2x2"],
    )
    def test_correctness(self, wpw, twg):
        cfg = _make_instance([1, 1, 1], wpw, twg, k_mult=4)
        _run_cdna4_gemm(cfg)


class TestCdna4Pipelined:
    """Pipelined K-loop sweep over all strategies, 4-wave symmetric."""

    @pytest.mark.parametrize("pipeline_strategy", list(range(11)))
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, pipeline_strategy, rotate_compute_stage):
        k_mult = max(4, _min_k_iters_for_ps(k_t=1, ps=pipeline_strategy))
        cfg = _make_instance(
            [1, 1, 1],
            [2, 2, 1],
            [2, 2, 1],
            k_mult,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_cdna4_gemm(cfg)


class TestCdna4Pipelined8Wave:
    """Pipelined K-loop with 8-wave configs and asymmetric strategies."""

    @pytest.mark.parametrize(
        "wpw,twg",
        [
            ([2, 4, 1], [8, 8, 2]),
            ([1, 8, 1], [8, 8, 1]),
        ],
        ids=["w2x4_twg8x8x2", "w1x8_twg8x8x1"],
    )
    @pytest.mark.parametrize("pipeline_strategy", [1, 2, 4, 6], ids=["ps1", "ps2", "ps4", "ps6"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, wpw, twg, pipeline_strategy, rotate_compute_stage):
        k_t = twg[DIM_K]
        min_iters = _min_k_iters_for_ps(k_t=k_t, ps=pipeline_strategy)
        # TODO(test/Dialect/AMDGCN/Transforms/cdna4-pipeliner-drain-fail.mlir):
        # for now, add a conservative +1 over minimum because
        #   k_iters == num_stages exactly
        # miscompiles in the drain epilogue for asymmetric strategies
        # (ps6 w1x8 twg8x8x1).
        k_mult = max(4, (min_iters + 1) * k_t)
        cfg = _make_instance(
            [1, 1, 1],
            wpw,
            twg,
            k_mult,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_cdna4_gemm(cfg)


class TestCdna4OperandPath:
    """Operand path sweep (direct_b only), orthogonal to geometry/pipeline."""

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([1, 1, 1], [2, 2, 1]),
            ([2, 2, 1], [2, 2, 1]),
        ],
        ids=["1w_2x2", "4w_2x2"],
    )
    @pytest.mark.parametrize("pipeline_strategy", [0, 3], ids=["ps0", "ps3"])
    @pytest.mark.parametrize("rotate_compute_stage", [False, True], ids=["norotc", "rotc"])
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg, pipeline_strategy, rotate_compute_stage):
        k_mult = max(4, _min_k_iters_for_ps(k_t=1, ps=pipeline_strategy))
        cfg = _make_instance(
            [1, 1, 1],
            num_waves_per_wg,
            num_tiles_per_wg,
            k_mult,
            pipeline_strategy=pipeline_strategy,
            rotate_compute_stage=rotate_compute_stage,
        )
        _run_cdna4_gemm(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--wg", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--wpw", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--twg", type=int, nargs=3, default=[2, 2, 1])
    parser.add_argument("--k-mult", type=int, default=4)
    parser.add_argument("--pipeline-strategy", type=int, default=0)
    args = parser.parse_args()

    cfg = _make_instance(
        args.wg,
        args.wpw,
        args.twg,
        args.k_mult,
        pipeline_strategy=args.pipeline_strategy,
    )
    gs = cfg.gemm_size
    print(
        f"Config: wg={'x'.join(map(str, args.wg))} wpw={'x'.join(map(str, args.wpw))} "
        f"twg={'x'.join(map(str, args.twg))} ps={args.pipeline_strategy}"
    )
    print(f"  M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}, waves={cfg.mapping.num_waves}")

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        _, asm = compile_cdna4_gemm(
            cfg,
            f.name,
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        )
    if args.print_asm:
        print(asm)
