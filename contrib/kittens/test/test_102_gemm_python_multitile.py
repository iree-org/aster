import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np
import pytest

from aster import ir
from aster.layout import Layout
from aster.dialects.kernel_builder_with_layouts import KernelBuilderWithLayouts as KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
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
    WeakScaledMappedGemmInstance,
)


def _build_multitile_gemm(cfg: "MultitileGemmInstance") -> ir.Module:
    """Build a multi-tile multi-wave multi-WG pipelined GEMM kernel."""
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
    lds_read_sub_tile_a = Layout((1, _exact_div(tile_k_elems, mfma_k, "tile_k/mfma_k")), (0, mfma_k * elt_bytes_a))
    lds_read_sub_tile_b = Layout((1, _exact_div(tile_k_elems, mfma_k, "tile_k/mfma_k")), (0, mfma_k * elt_bytes_b))
    lds_swizzle = mapping.lds_swizzle

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
        waves_k = max(1, num_waves // waves_s)
        if waves_s * waves_k != num_waves:
            waves_s = num_waves
            waves_k = 1
        coop_s = -(-num_tiles // waves_s)  # ceil div
        coop_k = -(-kt // waves_k)
        return waves_s, waves_k, coop_s, coop_k

    a_waves_m, a_waves_k, coop_a_m, coop_a_k = _coop_2d_split(twg_m, nw, k_t)
    b_waves_n, b_waves_k, coop_b_n, coop_b_k = _coop_2d_split(twg_n, nw, k_t)
    n_coop_a = coop_a_m * coop_a_k  # tiles loaded per wave for A
    n_coop_b = coop_b_n * coop_b_k  # tiles loaded per wave for B

    # Pipeline stage assignments from strategy.
    stg = PIPELINE_STRATEGIES[mapping.pipeline_strategy]
    STG_A_LOAD = stg["A_LOAD"]
    STG_A_LDS_WRITE = stg["A_LDS_WRITE"]
    STG_A_LDS_READ = stg["A_LDS_READ"]
    STG_B_LOAD = stg["B_LOAD"]
    STG_B_LDS_WRITE = stg["B_LDS_WRITE"]
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
    TILE_COORD_B = Layout((k_t, twg_n), (tile_row_bytes, mfma_n * stride_b))
    LDS_COORD_A = Layout((k_t, twg_m), (twg_m * tile_bytes, tile_bytes))
    LDS_COORD_B = Layout((k_t, twg_n), (twg_n * tile_bytes, tile_bytes))

    # Per-wave read coord: wave-local tile idx -> LDS byte offset relative to wave's LDS base.
    WAVE_READ_COORD_A = Layout((k_t, m_t), (twg_m * tile_bytes, tile_bytes))
    WAVE_READ_COORD_B = Layout((k_t, n_t), (twg_n * tile_bytes, tile_bytes))

    # WG base coord: (wg_idx, k_iter) -> global byte offset to WG's first tile.
    WG_BASE_A = Layout((wg[DIM_M], k_iters), (twg_m * TILE_COORD_A.strides[1], k_t * TILE_COORD_A.strides[0]))
    WG_BASE_B = Layout((wg[DIM_N], k_iters), (twg_n * TILE_COORD_B.strides[1], k_t * TILE_COORD_B.strides[0]))
    C_COORD = Layout((m_t, n_t), (mfma_m * stride_c_row, mfma_n * stride_c_col))

    # Distribution layouts: (wg_idx, wave_idx) -> global tile index.
    M_DIST = Layout((wg[DIM_M], wpw[DIM_M]), (twg_m, m_t))
    N_DIST = Layout((wg[DIM_N], wpw[DIM_N]), (twg_n, n_t))

    # Cooperative load: per-wave tile iteration + LDS write offset.
    # Each wave loads coop_m * coop_k tiles (A) or coop_n * coop_k tiles (B).
    # COOP_COORD maps per-wave tile idx -> global byte offset from WG base.
    COOP_COORD_A = Layout((coop_a_k, coop_a_m), (tile_row_bytes, mfma_m * stride_a))
    COOP_COORD_B = Layout((coop_b_k, coop_b_n), (tile_row_bytes, mfma_n * stride_b))
    # COOP_LDS maps per-wave tile idx -> LDS byte offset from wave's LDS base.
    COOP_LDS_A = Layout((coop_a_k, coop_a_m), (twg_m * tile_bytes, tile_bytes))
    COOP_LDS_B = Layout((coop_b_k, coop_b_n), (twg_n * tile_bytes, tile_bytes))
    # Wave load distribution: wave_id -> (m_start, k_start) in tile units.
    # OOB waves clamp to last valid start.
    max_a_m_start = max(0, twg_m - coop_a_m)
    max_a_k_start = max(0, k_t - coop_a_k)
    max_b_n_start = max(0, twg_n - coop_b_n)
    max_b_k_start = max(0, k_t - coop_b_k)

    n_tiles_a_wg, n_tiles_b_wg = k_t * twg_m, k_t * twg_n  # WG-level (for LDS)
    n_wtoks_per_tile = lds_write_sub_tile_a.size()
    n_frags_per_tile = lds_read_sub_tile_a.size()
    n_read_a, n_read_b = k_t * m_t, k_t * n_t
    lds_total_a = n_tiles_a_wg * tile_bytes
    lds_total_b = n_tiles_b_wg * tile_bytes

    d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)

    b = KernelBuilder("gemm_mt_mod", "gemm_multitile", target=mapping.mcpu, isa=mapping.isa)
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
    wave_m_idx, wave_n_idx = b.delinearize_index(b.wave_id(wave_size=ws), (wpw[DIM_M], wpw[DIM_N]))
    wid = b.wave_id(wave_size=ws)

    # Global tile-unit base for this wave (used for READ + C store).
    m_dist_idx = b.linearize_index((wg_m_idx, wave_m_idx), (wg[DIM_M], wpw[DIM_M]))
    n_dist_idx = b.linearize_index((wg_n_idx, wave_n_idx), (wg[DIM_N], wpw[DIM_N]))

    # Cooperative load starts: wave_id -> (m_start, k_start) with OOB clamping.
    ## A
    coop_a_m_start_raw = b.linearize_layout(wid, Layout((a_waves_m, a_waves_k), (coop_a_m, 0)))
    coop_a_k_start_raw = b.linearize_layout(wid, Layout((a_waves_m, a_waves_k), (0, coop_a_k)))
    coop_a_m_start = b.arith_minui(coop_a_m_start_raw, b.constant_index(max_a_m_start))
    coop_a_k_start = b.arith_minui(coop_a_k_start_raw, b.constant_index(max_a_k_start))
    ## B
    coop_b_n_start_raw = b.linearize_layout(wid, Layout((b_waves_n, b_waves_k), (coop_b_n, 0)))
    coop_b_k_start_raw = b.linearize_layout(wid, Layout((b_waves_n, b_waves_k), (0, coop_b_k)))
    coop_b_n_start = b.arith_minui(coop_b_n_start_raw, b.constant_index(max_b_n_start))
    coop_b_k_start = b.arith_minui(coop_b_k_start_raw, b.constant_index(max_b_k_start))

    c0, c1 = b.constant_index(0), b.constant_index(1)
    any_type = ir.Type.parse("!aster_utils.any")
    idx_type = ir.IndexType.get(b._ctx)

    # -- Scheduled helper functions for type erasure --
    # Preserved through constexpr expansion by selective-inlining (no
    # allow-scheduled-calls), inlined after pipelining by PHASE_SROA.
    sgpr2_type = ir.Type.parse("!amdgcn.sgpr<[? + 2]>")
    read_ret = [any_type, lds_read_tok] * n_frags_per_tile

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

    @b.define_helper("_read_a", [idx_type], read_ret)
    def read_a_fn(bb, lds_off):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_a, lds_swizzle, lds_read_sub_tile_a, bb.ds_read_b64
        )
        return [v for d, t in frags for v in (bb.to_any(d), t)]

    @b.define_helper("_read_b", [idx_type], read_ret)
    def read_b_fn(bb, lds_off):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_b, lds_swizzle, lds_read_sub_tile_b, bb.ds_read_b64
        )
        return [v for d, t in frags for v in (bb.to_any(d), t)]

    # -- Init accumulators in memref --
    c_buf = b.memref_alloca(b.constant_index(n_accs), ax4_type)

    @b.foreach_tile(n_accs)
    def _(idx):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, idx)

    # -- K-loop (void -- accumulators in c_buf) --
    @b.loop(c0, b.constant_index(k_iters), c1)
    def _(k_iv):
        # -- LDS ALLOC + LOAD (at LOAD stage -- pipeliner needs alloc in prologue) --
        with b.stage(STG_A_LOAD):
            lds_a_h, lds_a = b.alloc_lds(lds_total_a)
            lds_b_h, lds_b = b.alloc_lds(lds_total_b)

        # -- LOAD A (cooperative: scheduled func.call for type erasure) --
        with b.stage(STG_A_LOAD):
            a_wg_k_idx = b.linearize_index((wg_m_idx, k_iv), (wg[DIM_M], k_iters))
            a_wg_base = b.linearize_layout(a_wg_k_idx, WG_BASE_A)
            coop_a_off = b.linearize_layout(
                b.linearize_index((coop_a_k_start, coop_a_m_start), (k_t, twg_m)), TILE_COORD_A
            )
            a_wave_base = b.affine_apply(d0 + d1, [a_wg_base, coop_a_off])

            @b.foreach_tile(n_coop_a, types=[(any_type, 1), (flat_read_tok, 1)])
            def load_a(idx):
                tile_off = b.linearize_layout(idx, COOP_COORD_A)
                off = b.affine_apply(d0 + d1, [a_wave_base, tile_off])
                return b.call_helper(load_a_fn, [a_ptr, off], [any_type, flat_read_tok])

            data_buf_a, tok_buf_a = load_a

        # -- LOAD B (cooperative: scheduled func.call for type erasure) --
        with b.stage(STG_B_LOAD):
            b_wg_k_idx = b.linearize_index((wg_n_idx, k_iv), (wg[DIM_N], k_iters))
            b_wg_base = b.linearize_layout(b_wg_k_idx, WG_BASE_B)
            coop_b_off = b.linearize_layout(
                b.linearize_index((coop_b_k_start, coop_b_n_start), (k_t, twg_n)), TILE_COORD_B
            )
            b_wave_base = b.affine_apply(d0 + d1, [b_wg_base, coop_b_off])

            @b.foreach_tile(n_coop_b, types=[(any_type, 1), (flat_read_tok, 1)])
            def load_b(idx):
                tile_off = b.linearize_layout(idx, COOP_COORD_B)
                off = b.affine_apply(d0 + d1, [b_wave_base, tile_off])
                return b.call_helper(load_b_fn, [b_ptr, off], [any_type, flat_read_tok])

            data_buf_b, tok_buf_b = load_b

        # -- LDS WRITE A (scheduled func.call consumes any-typed data) --
        with b.stage(STG_A_LDS_WRITE):
            coop_a_lds_off = b.linearize_layout(
                b.linearize_index((coop_a_k_start, coop_a_m_start), (k_t, twg_m)), LDS_COORD_A
            )
            lds_a_wave = b.affine_apply(d0 + d1, [lds_a, coop_a_lds_off])

            @b.foreach_tile(n_coop_a, types=[(lds_write_tok, n_wtoks_per_tile)])
            def wtok_buf_a(idx):
                lds_off = b.affine_apply(d0 + d1, [lds_a_wave, b.linearize_layout(idx, COOP_LDS_A)])
                b.wait_deps(b.memref_load(tok_buf_a, idx))
                return b.call_helper(
                    write_a_fn,
                    [b.memref_load(data_buf_a, idx), lds_off],
                    [lds_write_tok] * n_wtoks_per_tile,
                )

        # -- LDS WRITE B (scheduled func.call consumes any-typed data) --
        with b.stage(STG_B_LDS_WRITE):
            coop_b_lds_off = b.linearize_layout(
                b.linearize_index((coop_b_k_start, coop_b_n_start), (k_t, twg_n)), LDS_COORD_B
            )
            lds_b_wave = b.affine_apply(d0 + d1, [lds_b, coop_b_lds_off])

            @b.foreach_tile(n_coop_b, types=[(lds_write_tok, n_wtoks_per_tile)])
            def wtok_buf_b(idx):
                lds_off = b.affine_apply(d0 + d1, [lds_b_wave, b.linearize_layout(idx, COOP_LDS_B)])
                b.wait_deps(b.memref_load(tok_buf_b, idx))
                return b.call_helper(
                    write_b_fn,
                    [b.memref_load(data_buf_b, idx), lds_off],
                    [lds_write_tok] * n_wtoks_per_tile,
                )

        # -- LDS READ A + DEALLOC (scheduled func.call returns any-typed frags) --
        with b.stage(STG_A_LDS_READ):

            @b.foreach_tile(n_coop_a * n_wtoks_per_tile)
            def _(i):
                b.wait_deps(b.memref_load(wtok_buf_a, i))

            b.s_barrier()

            wave_m_off = b.linearize_layout(wave_m_idx, Layout(wpw[DIM_M], m_t * tile_bytes))
            wave_lds_base_a = b.affine_apply(d0 + d1, [lds_a, wave_m_off])

            @b.foreach_tile(n_read_a, types=[(any_type, n_frags_per_tile), (lds_read_tok, n_frags_per_tile)])
            def read_a(idx):
                tile_off = b.linearize_layout(idx, WAVE_READ_COORD_A)
                off = b.affine_apply(d0 + d1, [wave_lds_base_a, tile_off])
                results = b.call_helper(read_a_fn, [off], read_ret)
                # Interleaved [any, tok, any, tok, ...] -> split into [any...] + [tok...]
                return [results[i] for i in range(0, len(results), 2)] + [results[i] for i in range(1, len(results), 2)]

            frag_buf_a, rtok_buf_a = read_a
            b.dealloc_lds(lds_a_h)

        # -- LDS READ B + DEALLOC --
        with b.stage(STG_B_LDS_READ):

            @b.foreach_tile(n_coop_b * n_wtoks_per_tile)
            def _(i):
                b.wait_deps(b.memref_load(wtok_buf_b, i))

            wave_n_off = b.linearize_layout(wave_n_idx, Layout(wpw[DIM_N], n_t * tile_bytes))
            wave_lds_base_b = b.affine_apply(d0 + d1, [lds_b, wave_n_off])

            @b.foreach_tile(n_read_b, types=[(any_type, n_frags_per_tile), (lds_read_tok, n_frags_per_tile)])
            def read_b(idx):
                tile_off = b.linearize_layout(idx, WAVE_READ_COORD_B)
                off = b.affine_apply(d0 + d1, [wave_lds_base_b, tile_off])
                results = b.call_helper(read_b_fn, [off], read_ret)
                return [results[i] for i in range(0, len(results), 2)] + [results[i] for i in range(1, len(results), 2)]

            frag_buf_b, rtok_buf_b = read_b
            b.dealloc_lds(lds_b_h)

        # -- COMPUTE (from_any to recover concrete types from any-typed frags) --
        with b.stage(STG_COMPUTE):

            @b.foreach_tile(m_t * n_t * k_t)
            def _(idx):
                mt, nt, kt = b.delinearize_index(idx, (m_t, n_t, k_t))
                acc_idx = b.linearize_index((mt, nt), (m_t, n_t))
                a_tile = b.linearize_index((kt, mt), (k_t, m_t))
                b_tile = b.linearize_index((kt, nt), (k_t, n_t))
                for sub in range(n_frags_per_tile):
                    a_fi = b.affine_apply(d0 * n_frags_per_tile + sub, [a_tile])
                    b_fi = b.affine_apply(d0 * n_frags_per_tile + sub, [b_tile])
                    b.wait_deps(b.memref_load(rtok_buf_a, a_fi), b.memref_load(rtok_buf_b, b_fi))
                    acc = b.memref_load(c_buf, acc_idx)
                    a_frag = b.from_any(b.memref_load(frag_buf_a, a_fi), vx2_type)
                    b_frag = b.from_any(b.memref_load(frag_buf_b, b_fi), vx2_type)
                    new_acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)
                    b.memref_store(new_acc, c_buf, acc_idx)

    # -- Store C tiles --
    m_base = b.linearize_layout(m_dist_idx, M_DIST)
    n_base = b.linearize_layout(n_dist_idx, N_DIST)
    total_m_tiles, total_n_tiles = wg[DIM_M] * twg_m, wg[DIM_N] * twg_n
    c_global_idx = b.linearize_index((m_base, n_base), (total_m_tiles, total_n_tiles))
    c_base = b.linearize_layout(c_global_idx, Layout((total_m_tiles, total_n_tiles), C_COORD.strides))

    @b.foreach_tile(n_accs)
    def _(idx):
        tile_off = b.linearize_layout(idx, C_COORD)
        c_off = b.affine_apply(d0 + d1, [c_base, tile_off])
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


def compile_multitile_gemm(cfg, output_hsaco_path, **kw):
    """Compile a multi-tile GEMM config to HSACO.

    Harness-compatible signature.
    """
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_multitile_gemm(cfg)
        pipeline = make_default_pass_pipeline(
            num_vgprs=kw.get("num_vgprs", 256),
            num_agprs=kw.get("num_agprs", 256),
            unroll_factor_multiplier=getattr(cfg.mapping, "unroll_factor_multiplier", 1),
            epilogue_peeling=getattr(cfg.mapping, "epilogue_peeling", True),
            ll_sched=getattr(cfg.mapping, "ll_sched", False),
            hoist_iter_arg_waits=getattr(cfg.mapping, "hoist_wait", False),
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

    Returns (C_output, times_ns).
    """
    from aster.execution.core import execute_hsaco, InputArray, OutputArray
    from aster.execution.utils import system_has_gpu

    mcpu = getattr(cfg.mapping, "mcpu", "gfx942")
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


def _make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy=1):
    """Build a MultitileGemmInstance from list parameters.

    M, N, K are derived from the tile grid and GemmSpec/GemmMappingSpec
    defaults (MFMA shape and transfer widths).
    """
    assert num_tiles_per_wg[DIM_M] % num_waves_per_wg[DIM_M] == 0, (
        f"twg_m({num_tiles_per_wg[DIM_M]}) not divisible by wpw_m({num_waves_per_wg[DIM_M]})"
    )
    assert num_tiles_per_wg[DIM_N] % num_waves_per_wg[DIM_N] == 0, (
        f"twg_n({num_tiles_per_wg[DIM_N]}) not divisible by wpw_n({num_waves_per_wg[DIM_N]})"
    )
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[
            num_tiles_per_wg[DIM_M] // num_waves_per_wg[DIM_M],
            num_tiles_per_wg[DIM_N] // num_waves_per_wg[DIM_N],
            num_tiles_per_wg[DIM_K],
        ],
        pipeline_strategy=pipeline_strategy,
    )
    # Derive M/N/K from tile grid using GemmSpec defaults for mfma_shape +
    # elt_bytes_a (and the real mapping's wave_size + global_load_bytes
    # which both match defaults here).
    # Match MultitileGemmInstance.transfer_tile_k_elems:
    #   (ws // mfma_m) * (xfer_bytes_a // elt_bytes_a).
    probe_spec = GemmSpec.from_sizes(1, 1, 1)
    mfma = probe_spec.mfma_shape
    tile_k_elems = (mapping.wave_size // mfma[DIM_M]) * (mapping.global_load_bytes // probe_spec.elt_bytes_a)
    M = num_workgroups[DIM_M] * num_tiles_per_wg[DIM_M] * mfma[DIM_M]
    N = num_workgroups[DIM_N] * num_tiles_per_wg[DIM_N] * mfma[DIM_N]
    k = k_mult * num_tiles_per_wg[DIM_K] * tile_k_elems
    return MultitileGemmInstance(GemmSpec.from_sizes(M, N, k), mapping)


def _run_multitile(cfg):
    """Compile + run a multi-tile GEMM, verify against numpy."""
    gs = cfg.gemm_size

    np.random.seed(42 + gs[DIM_M] + gs[DIM_N] + gs[DIM_K])
    A_mat = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
    B_mat = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)
    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_multitile_gemm(cfg)
        asm = compile_mlir_module_to_asm(module, pass_pipeline=make_default_pass_pipeline())

    mcpu = cfg.mapping.mcpu
    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

    with hsaco_file(path):
        if not system_has_mcpu(mcpu=mcpu):
            pytest.skip(f"{mcpu} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=KERNEL_NAME,
            arguments=[InputArray(A_mat.flatten()), InputArray(B_mat.flatten()), OutputArray(C_output)],
            grid_dim=(cfg.mapping.num_workgroups, 1, 1),
            block_dim=(cfg.mapping.num_threads, 1, 1),
        )

    expected = (A_mat.astype(np.float32) @ B_mat.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


def _min_k_iters(twg_k, ps):
    """Minimum K iterations for a given pipeline strategy."""
    from kittens_helpers import PIPELINE_STRATEGIES as PS

    return max(PS[ps].values()) + 1


class TestPythonGEMMMultitile:
    # Tile geometry: (num_workgroups, num_waves_per_wg, num_tiles_per_wg).
    # M = wg[M] * twg[M] * 16, N = wg[N] * twg[N] * 16.
    @pytest.mark.parametrize(
        "num_workgroups,num_waves_per_wg,num_tiles_per_wg",
        [
            # 1 wave
            ([1, 1, 1], [1, 1, 1], [3, 2, 1]),
            ([1, 1, 1], [1, 1, 1], [2, 2, 3]),  # deep K (k_t=3)
            ([1, 1, 1], [1, 1, 1], [4, 4, 1]),
            # 2 waves (2x1)
            ([1, 1, 1], [2, 1, 1], [4, 2, 1]),
            ([1, 1, 1], [2, 1, 1], [6, 2, 1]),
            # 4 waves (2x2)
            ([1, 1, 1], [2, 2, 1], [4, 4, 1]),
            ([1, 1, 1], [2, 2, 1], [8, 4, 1]),
            ([1, 1, 1], [2, 2, 1], [6, 4, 1]),  # non-power-of-2
            ([1, 1, 1], [2, 2, 1], [6, 6, 1]),  # non-power-of-2 both
            ([1, 1, 1], [2, 2, 1], [8, 8, 1]),  # larger tiles (sweep failure)
            # 4 waves (4x1)
            ([1, 1, 1], [4, 1, 1], [8, 7, 1]),
            ([1, 1, 1], [4, 1, 1], [12, 5, 1]),
            # 4 waves (1x4)
            ([1, 1, 1], [1, 4, 1], [10, 8, 1]),
            # 8 waves (4x2)
            ([1, 1, 1], [4, 2, 1], [8, 8, 1]),
            ([1, 1, 1], [4, 2, 1], [12, 6, 1]),
            # 8 waves (2x4)
            ([1, 1, 1], [2, 4, 1], [8, 8, 1]),
            ([1, 1, 1], [2, 4, 1], [12, 8, 1]),
            # Multi-WG
            ([3, 2, 1], [1, 1, 1], [3, 2, 1]),
            ([2, 2, 1], [2, 1, 1], [4, 2, 1]),
            ([2, 2, 1], [2, 2, 1], [4, 4, 1]),
        ],
        ids=[
            "1w_3x2",
            "1w_2x2x3_deepK",
            "1w_4x4",
            "2w_4x2",
            "2w_6x2",
            "4w_2x2_4x4",
            "4w_2x2_8x4",
            "4w_2x2_6x4_npow2",
            "4w_2x2_6x6_npow2",
            "4w_2x2_8x8",
            "4w_4x1_8x7",
            "4w_4x1_12x5",
            "4w_1x4_10x8",
            "8w_4x2_8x8",
            "8w_4x2_12x6",
            "8w_2x4_8x8",
            "8w_2x4_12x8",
            "mwg3x2_1w_3x2",
            "mwg2x2_2w_4x2",
            "mwg2x2_4w_4x4",
        ],
    )
    # K = k_mult * k_t * transfer_tile_k_elems: always divisible by construction.
    @pytest.mark.parametrize("k_mult", [2, 4, 8], ids=["km2", "km4", "km8"])
    @pytest.mark.parametrize("pipeline_strategy", [1, 3, 5], ids=["ps1", "ps3", "ps5"])
    def test_correctness(
        self,
        num_workgroups,
        num_waves_per_wg,
        num_tiles_per_wg,
        k_mult,
        pipeline_strategy,
    ):
        k_t = num_tiles_per_wg[DIM_K]
        if k_mult < _min_k_iters(k_t, pipeline_strategy):
            pytest.skip(f"k_mult={k_mult} < min_k_iters for ps{pipeline_strategy}")
        _run_multitile(_make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy))


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
