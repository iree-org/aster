"""CDNA4 (MI350, gfx950) sanity test on the non-G2S code path, direct_b operand_path only, mcpu=gfx950.

This is a carve-out of the TestCdna4Mfma class from
test_102_gemm_python_multitile.py specialized to the direct_b operand path.
It runs CDNA4 hardware through the cdna3 file's non-G2S (cooperative LDS-write
/ LDS-read) memory path for A while B bypasses LDS (per-wave global load at
preshuffle byte offsets). It serves as a sanity test that the common
KernelBuilder -- with vx4 join support but without direct global->LDS (G2S) --
works on CDNA4 hardware (gfx950) on the direct_b path.

Memory path: global_load_dwordx4 (A) -> ds_write_b64 -> ds_read_b64 (x2,
joined to vx4); B uses direct global_load_dwordx4 split into 2 vx2 fragments;
compute uses v_mfma_f32_16x16x32_f16.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np
import pytest

from aster import ir
import tempfile

from aster.layout import Layout, Swizzle, make_layout
from aster.dialects.kernel_builder_with_layouts import KernelBuilderWithLayouts as KernelBuilder
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


def _build_multitile_gemm(cfg: "MultitileGemmInstance", ping_pong_staggered: bool = False) -> ir.Module:
    """Build a multi-tile multi-wave multi-WG pipelined GEMM kernel.

    Specialized for direct_b: B bypasses LDS (per-wave global load at
    preshuffle byte offsets, vx4 split into 2 vx2 fragments for MFMA).

    Args:
      - ping_pong_staggered: triggers different mapping of copperative loads and
        per-wavegroup barrier staggering to enforce a ping-pong schedule.
    """
    from kittens_helpers import PIPELINE_STRATEGIES

    spec, mapping = cfg.spec, cfg.mapping
    gs = spec.gemm_size
    wg = mapping.num_workgroups_per_kernel
    wpw = mapping.num_waves_per_workgroup
    tpw = mapping.num_tiles_per_wave
    twg = mapping.num_tiles_per_workgroup
    ws = mapping.wave_size
    mfma_m, mfma_n, mfma_k = spec.mfma_shape[DIM_M], spec.mfma_shape[DIM_N], spec.mfma_shape[DIM_K]
    elt_bytes_a, elt_bytes_b = spec.elt_bytes_a, spec.elt_bytes_b

    # Per-transfer-tile geometry (derived from spec+mapping on cfg).
    tile_k_elems = cfg.transfer_tile_k_elems
    tile_row_bytes = cfg.transfer_tile_row_bytes
    tile_bytes = cfg.transfer_tile_bytes

    # Divisibility assertions -- no remainders allowed.
    def _exact_div(a, b, ctx=""):
        assert b != 0, f"division by zero: {ctx}"
        assert a % b == 0, f"{ctx}: {a} is not divisible by {b} (remainder {a % b})"
        return a // b

    # LDS tile layouts (derived from transfer widths and MFMA shape).
    lds_write_tile_a = Layout(
        (mfma_m, _exact_div(ws, mfma_m, "ws/mfma_m")), (tile_row_bytes, mapping.global_load_bytes)
    )
    lds_write_tile_b = Layout(
        (mfma_n, _exact_div(ws, mfma_n, "ws/mfma_n")), (tile_row_bytes, mapping.global_load_bytes)
    )
    lds_write_sub_tile_a = Layout(
        (1, _exact_div(mapping.global_load_bytes, mapping.ds_write_bytes, "xfer_a/ds_write")),
        (0, mapping.ds_write_bytes),
    )
    lds_write_sub_tile_b = Layout(
        (1, _exact_div(mapping.global_load_bytes, mapping.ds_write_bytes, "xfer_b/ds_write")),
        (0, mapping.ds_write_bytes),
    )
    lds_read_tile_a = Layout((_exact_div(ws, mfma_m, "ws/mfma_m"), mfma_m), (mapping.ds_read_bytes, tile_row_bytes))
    lds_read_tile_b = Layout((_exact_div(ws, mfma_n, "ws/mfma_n"), mfma_n), (mapping.ds_read_bytes, tile_row_bytes))
    # When the MFMA needs vx4 operands (CDNA4 16x16x32), each ds_read_b64
    # gives only half the K elements needed (vx2). Use frag_k = mfma_k // 2
    # so we get 2 fragments per tile, joined to vx4 in the compute loop.
    from aster.dialects.kernel_builder import MFMA_F16_CDNA4

    # cdna4 (vx4 join) MFMA: each ds_read_b64 gives half the K elements; 2 fragments joined per MFMA.
    assert mfma_k == MFMA_F16_CDNA4.shape[2], f"cdna4 leaf requires mfma_k={MFMA_F16_CDNA4.shape[2]}, got {mfma_k}"
    frag_k = mfma_k // 2
    lds_read_sub_tile_a = Layout((1, _exact_div(tile_k_elems, frag_k, "tile_k/frag_k")), (0, frag_k * elt_bytes_a))
    lds_read_sub_tile_b = Layout((1, _exact_div(tile_k_elems, frag_k, "tile_k/frag_k")), (0, frag_k * elt_bytes_b))
    lds_swizzle = Swizzle(bits=3, base=3, shift=4)

    m_t, n_t, k_t = tpw[DIM_M], tpw[DIM_N], tpw[DIM_K]
    twg_m, twg_n = twg[DIM_M], twg[DIM_N]
    nw = mapping.num_waves

    # Tile/wave divisibility.
    assert twg_m == wpw[DIM_M] * m_t, f"twg_m({twg_m}) != wpw_m({wpw[DIM_M]}) * m_t({m_t})"
    assert twg_n == wpw[DIM_N] * n_t, f"twg_n({twg_n}) != wpw_n({wpw[DIM_N]}) * n_t({n_t})"

    # 2-D cooperative split: each wave loads coop_m * coop_k A tiles
    # and coop_n * coop_k B tiles (instead of ALL twg tiles).
    # When nw doesn't factor as waves_s * waves_k, all waves go spatial
    # and excess waves duplicate the last tile via arith_minui clamping.
    def _coop_2d_split(num_tiles, num_waves, kt):
        waves_s = min(num_tiles, num_waves)
        waves_k = max(1, math.floor(num_waves / waves_s))
        coop_s = math.ceil(num_tiles / waves_s)
        coop_k = math.ceil(kt / waves_k)
        return waves_s, waves_k, coop_s, coop_k

    # Per-group cooperative loading: when staggered, each half-WG (4 waves)
    # independently loads ALL tiles so the stagger doesn't leave partial data.
    nw_coop = nw // 2 if ping_pong_staggered else nw
    a_waves_m, a_waves_k, coop_a_m, coop_a_k = _coop_2d_split(twg_m, nw_coop, k_t)
    n_coop_a = coop_a_m * coop_a_k  # tiles loaded per wave for A

    # Pipeline stage assignments from strategy.
    stg = PIPELINE_STRATEGIES[mapping.pipeline_strategy]
    STG_A_LOAD = stg["A_LOAD"]
    STG_A_LDS_WRITE = stg["A_LDS_WRITE"]
    STG_A_LDS_READ = stg["A_LDS_READ"]
    STG_B_LOAD = stg["B_LOAD"]
    STG_B_LDS_READ = stg["B_LDS_READ"]
    STG_COMPUTE = stg["COMPUTE"]
    ol_a, ol_b, ol_c = spec.operand_layout(OP_A), spec.operand_layout(OP_B), spec.operand_layout(OP_C)
    stride_a, stride_b = ol_a.strides[0], ol_b.strides[0]
    stride_c_row, stride_c_col = ol_c.strides[0], ol_c.strides[1]

    k_step = k_t * tile_k_elems
    assert gs[DIM_K] % k_step == 0, f"K={gs[DIM_K]} must be divisible by k_t*tile_k_elems={k_step}"
    k_iters = gs[DIM_K] // k_step
    n_accs = m_t * n_t

    # Global load layouts (stride-dependent).
    GLOBAL_LOAD_TILE_A = Layout((mfma_m, ws // mfma_m), (stride_a, mapping.global_load_bytes))
    GLOBAL_LOAD_SUB_TILE_A = Layout(1, 0)
    GLOBAL_LOAD_TILE_B = Layout((mfma_n, ws // mfma_n), (stride_b, mapping.global_load_bytes))
    GLOBAL_LOAD_SUB_TILE_B = Layout(1, 0)

    # Global store layout.
    n_agprs = ws // mfma_n
    GLOBAL_STORE_TILE_C = Layout((n_agprs, mfma_n, n_agprs), (n_agprs * stride_c_row, stride_c_col, stride_c_row))
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    # Tile coord layouts: (k_tile, parallel_tile) -> byte offset.
    TILE_COORD_A = Layout((k_t, twg_m), (tile_row_bytes, mfma_m * stride_a))
    LDS_COORD_A = Layout((k_t, twg_m), (twg_m * tile_bytes, tile_bytes))

    # Per-wave read coord: wave-local tile idx -> LDS byte offset relative to wave's LDS base.
    WAVE_READ_COORD_A = Layout((k_t, m_t), (twg_m * tile_bytes, tile_bytes))
    # Flat wave_id -> wpw_m / wpw_n -> M-only byte stride.
    WAVE_M_LDS_OFF = Layout((wpw[DIM_M], wpw[DIM_N]), (m_t * tile_bytes, 0))

    # WG base coord: (wg_idx, k_iter) -> global byte offset to WG's first tile.
    WG_BASE_A = Layout((wg[DIM_M], k_iters), (twg_m * TILE_COORD_A.strides[1], k_t * TILE_COORD_A.strides[0]))

    # Per-wave A base layout via make_layout (fused WG_BASE_A + TILE_COORD_A).
    WAVE_BASE_A = make_layout(WG_BASE_A, TILE_COORD_A)
    C_COORD = Layout((m_t, n_t), (mfma_m * stride_c_row, mfma_n * stride_c_col))

    # Global C-tile base in bytes: (wg_m_idx, wave_m_idx, wg_n_idx, wave_n_idx)
    # -> byte offset of this wave's first C tile.
    C_BASE = Layout(
        sizes=((wg[DIM_M], wpw[DIM_M]), (wg[DIM_N], wpw[DIM_N])),
        strides=(
            (twg_m * mfma_m * stride_c_row, m_t * mfma_m * stride_c_row),
            (twg_n * mfma_n * stride_c_col, n_t * mfma_n * stride_c_col),
        ),
    )
    # Per-thread C-tile byte offset = C_BASE + per-tile (m, n) offset.
    C_OFF = make_layout(C_BASE, C_COORD)

    # Preshuffle B layout: (n_block, k_block, lane_id) -> byte offset.
    # Matches shuffle_weight() in kittens_helpers.py.
    _, kb = cfg.preshuffle_n_blocks, cfg.preshuffle_k_blocks
    lane_s, k_s = cfg.preshuffle_lane_stride_bytes, cfg.preshuffle_k_block_stride_bytes
    stride_n0_bytes = kb * k_s
    # Preshuffle B byte offset as a single rank-3 nested layout over the 6
    # atomic coords (wg_n_idx, wave_n_idx, nt, k_iv, kt, lid). Fuses N_DIST,
    # the per-iter (n_block, k_block) math, and the original PRESHUFFLE_LAYOUT.
    PRESHUFFLE_FULL = Layout(
        sizes=((wg[DIM_N], wpw[DIM_N], n_t), (k_iters, k_t), ws),
        strides=(
            (twg_n * stride_n0_bytes, n_t * stride_n0_bytes, stride_n0_bytes),
            (k_t * k_s, k_s),
            lane_s,
        ),
    )

    # Cooperative load: per-wave tile iteration + LDS write offset.
    # Each wave loads coop_m * coop_k tiles for A. (B is direct-loaded; no LDS.)
    # COOP_COORD maps per-wave tile idx -> global byte offset from WG base.
    COOP_COORD_A = Layout((coop_a_k, coop_a_m), (tile_row_bytes, mfma_m * stride_a))
    # COOP_LDS maps per-wave tile idx -> LDS byte offset from wave's LDS base.
    COOP_LDS_A = Layout((coop_a_k, coop_a_m), (twg_m * tile_bytes, tile_bytes))
    # Wave load distribution: wave_id -> (m_start, k_start) in tile units.
    # OOB waves clamp to last valid start.
    max_a_m_start = max(0, twg_m - coop_a_m)
    max_a_k_start = max(0, k_t - coop_a_k)

    n_tiles_a_wg = k_t * twg_m  # WG-level (for LDS)
    n_wtoks_per_tile = lds_write_sub_tile_a.size()
    n_frags_per_tile = lds_read_sub_tile_a.size()
    n_read_a, n_read_b = k_t * m_t, k_t * n_t
    lds_total_a = n_tiles_a_wg * tile_bytes

    d0 = ir.AffineExpr.get_dim(0)

    b = KernelBuilder("gemm_mod", cfg.kernel_name, target=mapping.mcpu)
    b.set_block_dims(mapping.num_threads)
    b.set_grid_dims(mapping.num_workgroups)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # -- Register/token types for memref buffers --
    from aster._mlir_libs._amdgcn import AGPRRangeType, VGPRRangeType

    vx4_type = VGPRRangeType.get(b._ctx, size=4)
    vx2_type = VGPRRangeType.get(b._ctx, size=2)
    ax4_type = AGPRRangeType.get(b._ctx, size=4)
    flat_read_tok = ir.Type.parse("!amdgcn.read_token<flat>")
    lds_write_tok = ir.Type.parse("!amdgcn.write_token<shared>")
    lds_read_tok = ir.Type.parse("!amdgcn.read_token<shared>")

    # -- Distribution --
    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg[DIM_M], wg[DIM_N]))
    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw[DIM_M], wpw[DIM_N]))

    # Cooperative load starts: wave_id -> (m_start, k_start) with OOB clamping.
    # When staggered, use local wave_id within 4-wave group.
    coop_wid = b.affine_apply(d0 % nw_coop, [wid]) if ping_pong_staggered else wid
    coop_a_m_raw, coop_a_k_raw = b.delinearize_index(coop_wid, (a_waves_m, a_waves_k))
    coop_a_m_start = b.arith_minui(b.affine_apply(d0 * coop_a_m, [coop_a_m_raw]), b.constant_index(max_a_m_start))
    coop_a_k_start = b.arith_minui(b.affine_apply(d0 * coop_a_k, [coop_a_k_raw]), b.constant_index(max_a_k_start))

    c0, c1 = b.constant_index(0), b.constant_index(1)
    any_type = ir.Type.parse("!aster_utils.any")
    idx_type = ir.IndexType.get(b._ctx)

    # -- Scheduled helper functions for type erasure --
    # Preserved through constexpr expansion by selective-inlining (no
    # allow-scheduled-calls), inlined after pipelining by PHASE_SROA.
    sgpr2_type = ir.Type.parse("!amdgcn.sgpr<[? + 2]>")
    read_ret = [any_type] * n_frags_per_tile + [lds_read_tok] * n_frags_per_tile

    @b.define_helper("_load_a", [sgpr2_type, idx_type], [any_type, flat_read_tok])
    def load_a_fn(bb, ptr, off):
        [(d, t)] = bb.load_multi_tile_from_global(
            ptr, off, GLOBAL_LOAD_TILE_A, GLOBAL_LOAD_SUB_TILE_A, bb.global_load_dwordx4
        )
        return [bb.to_any(d), t]

    @b.define_helper("_load_b", [sgpr2_type, idx_type], [any_type, flat_read_tok])
    def load_b_fn(bb, ptr, off):
        [(d, t)] = bb.load_multi_tile_from_global(
            ptr, off, GLOBAL_LOAD_TILE_B, GLOBAL_LOAD_SUB_TILE_B, bb.global_load_dwordx4
        )
        return [bb.to_any(d), t]

    @b.define_helper("_load_b_direct", [sgpr2_type, idx_type], [any_type, flat_read_tok])
    def load_b_direct_fn(bb, ptr, byte_off):
        d, t = bb.global_load_dwordx4(ptr, dynamic_offset=bb.index_to_vgpr(byte_off))
        return [bb.to_any(d), t]

    @b.define_helper("_write_a", [any_type, idx_type], [lds_write_tok] * n_wtoks_per_tile)
    def write_a_fn(bb, data_any, lds_off):
        return bb.write_multi_tile_to_lds(
            bb.from_any(data_any, vx4_type),
            lds_off,
            lds_write_tile_a,
            lds_swizzle,
            lds_write_sub_tile_a,
            bb.ds_write_b64,
        )

    @b.define_helper("_write_b", [any_type, idx_type], [lds_write_tok] * n_wtoks_per_tile)
    def write_b_fn(bb, data_any, lds_off):
        return bb.write_multi_tile_to_lds(
            bb.from_any(data_any, vx4_type),
            lds_off,
            lds_write_tile_b,
            lds_swizzle,
            lds_write_sub_tile_b,
            bb.ds_write_b64,
        )

    @b.define_helper("_read_a", [idx_type, b.fence_tok], read_ret)
    def read_a_fn(bb, lds_off, fence):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_a, lds_swizzle, lds_read_sub_tile_a, bb.ds_read_b64, fence_token=fence
        )
        return [bb.to_any(d) for d, t in frags] + [t for d, t in frags]

    @b.define_helper("_read_b", [idx_type], read_ret)
    def read_b_fn(bb, lds_off):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_b, lds_swizzle, lds_read_sub_tile_b, bb.ds_read_b64
        )
        return [bb.to_any(d) for d, t in frags] + [t for d, t in frags]

    # -- Init accumulators in memref --
    c_buf = b.memref_alloca(b.constant_index(n_accs), ax4_type)

    @b.foreach_tile(n_accs)
    def _(idx):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, idx)

    if ping_pong_staggered:

        @b.thread_uniform_if("ult", wid, b.constant_index(4))
        def _():
            b.s_barrier()

    # -- K-loop (void -- accumulators in c_buf) --
    @b.loop(c0, b.constant_index(k_iters), c1)
    def _(k_iv):
        # -- LDS ALLOC --
        # Allocate at the earliest LOAD stage for max buffer distance.
        # Only A is allocated (direct_b: B bypasses LDS).
        with b.stage(STG_A_LOAD):
            lds_a_h, lds_a = b.alloc_lds(lds_total_a)

        # -- LOAD A (cooperative: scheduled func.call for type erasure) --
        with b.stage(STG_A_LOAD):
            a_wave_base = b.layout_apply((wg_m_idx, k_iv, coop_a_k_start, coop_a_m_start), WAVE_BASE_A)

            @b.foreach_tile(n_coop_a, types=[(any_type, 1), (flat_read_tok, 1)])
            def load_a(idx):
                tile_off = b.layout_apply(idx, COOP_COORD_A)
                off = b.layout_sum(a_wave_base, tile_off)
                return b.call_helper(load_a_fn, [a_ptr, off], [any_type, flat_read_tok])

            data_buf_a, tok_buf_a = load_a

        # -- LOAD B (direct: per-wave load at preshuffle byte offsets) --
        with b.stage(STG_B_LOAD):
            lid = b.lane_id()

            @b.foreach_tile(k_t * n_t, types=[(any_type, 1), (flat_read_tok, 1)])
            def load_b(idx):
                kt, nt = b.delinearize_index(idx, (k_t, n_t))
                byte_off = b.layout_apply((wg_n_idx, wave_n_idx, nt, k_iv, kt, lid), PRESHUFFLE_FULL)
                return b.call_helper(load_b_direct_fn, [b_ptr, byte_off], [any_type, flat_read_tok])

            data_buf_b, tok_buf_b = load_b

        # -- LDS WRITE A --
        with b.stage(STG_A_LDS_WRITE):
            coop_a_lds_off = b.layout_apply((coop_a_k_start, coop_a_m_start), LDS_COORD_A)
            lds_a_wave = b.layout_sum(lds_a, coop_a_lds_off)

            @b.foreach_tile(n_coop_a, types=[(lds_write_tok, n_wtoks_per_tile)])
            def wtok_buf_a(idx):
                lds_off = b.layout_sum(lds_a_wave, b.layout_apply(idx, COOP_LDS_A))
                b.wait_deps(b.memref_load(tok_buf_a, idx))
                return b.call_helper(
                    write_a_fn,
                    [b.memref_load(data_buf_a, idx), lds_off],
                    [lds_write_tok] * n_wtoks_per_tile,
                )

        # -- LDS READ A + DEALLOC --
        with b.stage(STG_A_LDS_READ):
            a_wtoks = [b.memref_load(wtok_buf_a, b.constant_index(i)) for i in range(n_coop_a * n_wtoks_per_tile)]
            b.wait_deps(*a_wtoks)
            bfence = b.cross_wave_token_barrier(*a_wtoks)
            wave_m_off = b.layout_apply(wid, WAVE_M_LDS_OFF)
            wave_lds_base_a = b.layout_sum(lds_a, wave_m_off)

            @b.foreach_tile(n_read_a, types=[(any_type, n_frags_per_tile), (lds_read_tok, n_frags_per_tile)])
            def read_a(idx):
                tile_off = b.layout_apply(idx, WAVE_READ_COORD_A)
                off = b.layout_sum(wave_lds_base_a, tile_off)
                return b.call_helper(read_a_fn, [off, bfence], read_ret)

            frag_buf_a, rtok_buf_a = read_a
            b.dealloc_lds(lds_a_h)

        # -- B FRAGMENTS: wait global loads, split vx4 -> 2*vx2 fragments. --
        with b.stage(STG_B_LDS_READ):

            @b.foreach_tile(n_read_b, types=[(any_type, n_frags_per_tile), (flat_read_tok, n_frags_per_tile)])
            def read_b(idx):
                tok = b.memref_load(tok_buf_b, idx)
                b.wait_deps(tok)
                b_vx4 = b.from_any(b.memref_load(data_buf_b, idx), vx4_type)
                b_lo, b_hi = b.split_vx4(b_vx4)
                return [b.to_any(b_lo), b.to_any(b_hi)] + [tok] * n_frags_per_tile

        frag_buf_b, rtok_buf_b = read_b

        # -- COMPUTE --
        with b.stage(STG_COMPUTE):
            nf = n_frags_per_tile

            # Coarse wait at the top of the COMPUTE stage give more opportunities
            # to llsched.
            all_a_tokens = [b.memref_load(rtok_buf_a, b.constant_index(i)) for i in range(k_t * m_t * nf)]
            all_b_tokens = [b.memref_load(rtok_buf_b, b.constant_index(i)) for i in range(k_t * n_t * nf)]
            b.wait_deps(*all_a_tokens, *all_b_tokens)

            # CDNA4: join 2 vx2 fragments -> vx4 per MFMA.
            assert nf == 2, f"CDNA4 expects 2 vx2 frags per tile, got {nf}"

            @b.foreach_tile(k_t * m_t * n_t)
            def _(idx):
                kt, mt, nt = b.delinearize_index(idx, (k_t, m_t, n_t))
                acc_idx = b.linearize_index((mt, nt), (m_t, n_t))
                a_lo_i = b.linearize_index((kt, mt, b.constant_index(0)), (k_t, m_t, nf))
                a_hi_i = b.linearize_index((kt, mt, b.constant_index(1)), (k_t, m_t, nf))
                b_lo_i = b.linearize_index((kt, nt, b.constant_index(0)), (k_t, n_t, nf))
                b_hi_i = b.linearize_index((kt, nt, b.constant_index(1)), (k_t, n_t, nf))
                acc = b.memref_load(c_buf, acc_idx)
                a_vx4 = b.join_vx2_to_vx4(
                    b.from_any(b.memref_load(frag_buf_a, a_lo_i), vx2_type),
                    b.from_any(b.memref_load(frag_buf_a, a_hi_i), vx2_type),
                )
                b_vx4 = b.join_vx2_to_vx4(
                    b.from_any(b.memref_load(frag_buf_b, b_lo_i), vx2_type),
                    b.from_any(b.memref_load(frag_buf_b, b_hi_i), vx2_type),
                )
                new_acc = b.mfma(MFMA_F16_CDNA4.opcode, acc, a_vx4, b_vx4)
                b.memref_store(new_acc, c_buf, acc_idx)

            if ping_pong_staggered:
                b.s_barrier()

    if ping_pong_staggered:

        @b.thread_uniform_if("uge", wid, b.constant_index(4))
        def _():
            b.s_barrier()

    # -- Store C tiles --
    @b.foreach_tile(n_accs)
    def _(idx):
        m, n = b.delinearize_index(idx, (m_t, n_t))
        c_off = b.layout_apply((wg_m_idx, wave_m_idx, wg_n_idx, wave_n_idx, m, n), C_OFF)
        acc = b.memref_load(c_buf, idx)
        b.store_multi_fragment_to_global(
            acc, c_ptr, c_off, GLOBAL_STORE_TILE_C, GLOBAL_STORE_SUB_TILE_C, b.global_store_dword
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

    Preshuffles B (direct_b operand path). Returns (C_output, times_ns).
    """
    from kittens_helpers import shuffle_weight

    mcpu = getattr(cfg.mapping, "mcpu", "gfx942")
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
    """LL-scheduler preset axis (ll_sched 0..5) on a register-safe 4w geometry, gfx950.

    0 = scheduler off; 1 = default mfma-hiding preset; 3/5 engage the
    deterministic xdlMaxRun interleaving cap (3 = mid, 5 =
    max-1-contiguous).
    """

    @pytest.mark.parametrize("ll_sched", [0, 1, 3, 5], ids=lambda v: f"llsched{v}")
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
