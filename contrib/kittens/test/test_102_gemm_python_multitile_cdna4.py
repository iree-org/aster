"""CDNA4 (MI350, gfx950) multi-tile GEMM using KernelBuilderWithLayouts.

Same structure as test_102_gemm_python_multitile but targeting CDNA4:
  - v_mfma_f32_16x16x32_f16 (doubled-K, 4 VGPR A/B operands)
  - ds_read_b128 (128-bit LDS reads for vx4 MFMA fragments)
  - 1 fragment per tile (mfma_k == tile_k_elems, no sub-tiling)

Memory path: G2S buffer_load_dwordx4_lds -> ds_read_b64 (2x, joined to vx4) -> MFMA.
All tiles loaded cooperatively by all waves, barrier, then all waves compute.
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
from aster.dialects.kernel_builder_with_layouts import KernelBuilderWithLayouts as KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_gpu
from aster.pass_pipelines import make_default_pass_pipeline
from aster.layout import Layout, Swizzle

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


def _build_cdna4_gemm(cfg: "Cdna4GemmInstance") -> ir.Module:
    """Build a CDNA4 multi-tile GEMM kernel via G2S direct-to-LDS.

    Same distribution pattern as test_102 (multi-WG + per-wave tile
    strips), with G2S leader loads and wg barrier; no direct_b, ping-
    pong, or pipelining atm.
    """
    spec, mapping = cfg.spec, cfg.mapping
    gs = spec.gemm_size
    wg = mapping.num_workgroups_per_kernel
    wpw = mapping.num_waves_per_workgroup
    tpw = mapping.num_tiles_per_wave
    twg = mapping.num_tiles_per_workgroup
    ws = mapping.wave_size
    mfma_m, mfma_n, mfma_k = spec.mfma_shape[DIM_M], spec.mfma_shape[DIM_N], spec.mfma_shape[DIM_K]
    elt_bytes_a, elt_bytes_b = spec.elt_bytes_a, spec.elt_bytes_b

    tile_k_elems = cfg.transfer_tile_k_elems
    tile_row_bytes = cfg.transfer_tile_row_bytes
    tile_bytes = cfg.transfer_tile_bytes

    def _exact_div(a, b, ctx=""):
        assert b != 0, f"division by zero: {ctx}"
        assert a % b == 0, f"{ctx}: {a} is not divisible by {b}"
        return a // b

    m_t, n_t, k_t = tpw[DIM_M], tpw[DIM_N], tpw[DIM_K]
    twg_m, twg_n = twg[DIM_M], twg[DIM_N]
    nw = mapping.num_waves
    assert twg_m == wpw[DIM_M] * m_t
    assert twg_n == wpw[DIM_N] * n_t

    # 2-D cooperative split (same as test_102): distribute WG tiles across waves.
    def _coop_2d_split(num_tiles, num_waves, kt):
        waves_s = min(num_tiles, num_waves)
        waves_k = max(1, math.floor(num_waves / waves_s))
        coop_s = math.ceil(num_tiles / waves_s)
        coop_k = math.ceil(kt / waves_k)
        return waves_s, waves_k, coop_s, coop_k

    a_waves_m, a_waves_k, coop_a_m, coop_a_k = _coop_2d_split(twg_m, nw, k_t)
    b_waves_n, b_waves_k, coop_b_n, coop_b_k = _coop_2d_split(twg_n, nw, k_t)
    n_coop_a = coop_a_m * coop_a_k
    n_coop_b = coop_b_n * coop_b_k

    ol_a, ol_b, ol_c = spec.operand_layout(OP_A), spec.operand_layout(OP_B), spec.operand_layout(OP_C)
    stride_a, stride_b = ol_a.strides[0], ol_b.strides[0]
    stride_c_row, stride_c_col = ol_c.strides[0], ol_c.strides[1]

    k_step = k_t * tile_k_elems
    assert gs[DIM_K] % k_step == 0
    k_iters = gs[DIM_K] // k_step
    n_accs = m_t * n_t

    # -- Layouts --

    lds_swizzle = Swizzle(bits=2, base=4, shift=4)
    frag_k = mfma_k // 2  # K elements per fragment = 16 (half-K)
    lds_read_tile_a = Layout((_exact_div(ws, mfma_m, "ws/mfma_m"), mfma_m), (mapping.ds_read_bytes, tile_row_bytes))
    lds_read_tile_b = Layout((_exact_div(ws, mfma_n, "ws/mfma_n"), mfma_n), (mapping.ds_read_bytes, tile_row_bytes))
    n_frags_per_tile = _exact_div(tile_k_elems, frag_k, "tile_k/frag_k")
    lds_read_sub_tile_a = Layout((1, n_frags_per_tile), (0, frag_k * elt_bytes_a))
    lds_read_sub_tile_b = Layout((1, n_frags_per_tile), (0, frag_k * elt_bytes_b))

    # Per-thread global byte offset within one tile, for the G2S voffset.
    #
    # G2S (`buffer_load_dwordx4_lds`) always writes `M0 + TID*16` linearly to LDS.
    # Swizzle acts on tile-local bits only: compute the tile-local offset with
    # `tile_row_bytes` row stride (swizzled), then add the row-stride correction
    # `(stride_a - tile_row_bytes, 0)` unswizzled.
    TILE_LOCAL_A = Layout((mfma_m, ws // mfma_m), (tile_row_bytes, mapping.global_load_bytes))
    TILE_LOCAL_B = Layout((mfma_n, ws // mfma_n), (tile_row_bytes, mapping.global_load_bytes))
    ROW_STRIDE_CORR_A = Layout((mfma_m, ws // mfma_m), (stride_a - tile_row_bytes, 0))
    ROW_STRIDE_CORR_B = Layout((mfma_n, ws // mfma_n), (stride_b - tile_row_bytes, 0))

    n_agprs = ws // mfma_n
    GLOBAL_STORE_TILE_C = Layout((n_agprs, mfma_n, n_agprs), (n_agprs * stride_c_row, stride_c_col, stride_c_row))
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    TILE_COORD_A = Layout((k_t, twg_m), (tile_row_bytes, mfma_m * stride_a))
    TILE_COORD_B = Layout((k_t, twg_n), (tile_row_bytes, mfma_n * stride_b))
    LDS_COORD_A = Layout((k_t, twg_m), (twg_m * tile_bytes, tile_bytes))
    LDS_COORD_B = Layout((k_t, twg_n), (twg_n * tile_bytes, tile_bytes))
    WAVE_READ_COORD_A = Layout((k_t, m_t), (twg_m * tile_bytes, tile_bytes))
    WAVE_READ_COORD_B = Layout((k_t, n_t), (twg_n * tile_bytes, tile_bytes))
    WAVE_M_LDS_OFF = Layout((wpw[DIM_M], wpw[DIM_N]), (m_t * tile_bytes, 0))
    WAVE_N_LDS_OFF = Layout((wpw[DIM_M], wpw[DIM_N]), (0, n_t * tile_bytes))
    WG_BASE_A = Layout((wg[DIM_M], k_iters), (twg_m * TILE_COORD_A.strides[1], k_t * TILE_COORD_A.strides[0]))
    WG_BASE_B = Layout((wg[DIM_N], k_iters), (twg_n * TILE_COORD_B.strides[1], k_t * TILE_COORD_B.strides[0]))
    M_DIST = Layout((wg[DIM_M], wpw[DIM_M]), (twg_m, m_t))
    N_DIST = Layout((wg[DIM_N], wpw[DIM_N]), (twg_n, n_t))
    C_COORD = Layout((m_t, n_t), (mfma_m * stride_c_row, mfma_n * stride_c_col))

    # Cooperative load layouts (same pattern as test_102).
    COOP_COORD_A = Layout((coop_a_k, coop_a_m), (tile_row_bytes, mfma_m * stride_a))
    COOP_COORD_B = Layout((coop_b_k, coop_b_n), (tile_row_bytes, mfma_n * stride_b))
    COOP_LDS_A = Layout((coop_a_k, coop_a_m), (twg_m * tile_bytes, tile_bytes))
    COOP_LDS_B = Layout((coop_b_k, coop_b_n), (twg_n * tile_bytes, tile_bytes))
    max_a_m_start = max(0, twg_m - coop_a_m)
    max_a_k_start = max(0, k_t - coop_a_k)
    max_b_n_start = max(0, twg_n - coop_b_n)
    max_b_k_start = max(0, k_t - coop_b_k)

    n_tiles_a_wg = k_t * twg_m
    n_tiles_b_wg = k_t * twg_n
    n_read_a, n_read_b = k_t * m_t, k_t * n_t
    lds_total_a = n_tiles_a_wg * tile_bytes
    lds_total_b = n_tiles_b_wg * tile_bytes

    d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)

    b = KernelBuilder("gemm_cdna4_mod", cfg.kernel_name, target=mapping.mcpu, isa=mapping.isa)
    b.set_block_dims(mapping.num_threads)
    b.set_grid_dims(mapping.num_workgroups)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # Buffer resource descriptors for G2S.
    SRD_FLAGS = 0x24924
    nr_a = b.s_mov_b32(gs[DIM_M] * stride_a)
    nr_b = b.s_mov_b32(gs[DIM_N] * stride_b)
    stride_i32 = b.constant_i32(0)
    a_rsrc = b.make_buffer_rsrc(a_ptr, nr_a, stride_i32, flags=SRD_FLAGS)
    b_rsrc = b.make_buffer_rsrc(b_ptr, nr_b, stride_i32, flags=SRD_FLAGS)
    soff = b.s_mov_b32(0)
    m0 = b.alloc_m0()
    soff_val = soff.result

    # Per-thread byte offsets within one tile (loop-invariant).
    lane = b.lane_id()
    thread_off_a = b.affine_apply(
        d0 + d1,
        [
            b.linearize_layout(lane, TILE_LOCAL_A, swizzle=lds_swizzle),
            b.linearize_layout(lane, ROW_STRIDE_CORR_A),
        ],
    )
    thread_off_b = b.affine_apply(
        d0 + d1,
        [
            b.linearize_layout(lane, TILE_LOCAL_B, swizzle=lds_swizzle),
            b.linearize_layout(lane, ROW_STRIDE_CORR_B),
        ],
    )

    # LDS read helpers.
    read_ret = [b.any_type] * n_frags_per_tile + [b.lds_read_tok] * n_frags_per_tile

    @b.define_helper("_read_a", [b.idx_type], read_ret)
    def read_a_fn(bb, lds_off):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_a, lds_swizzle, lds_read_sub_tile_a, bb.ds_read_b64
        )
        return [bb.to_any(d) for d, t in frags] + [t for d, t in frags]

    @b.define_helper("_read_b", [b.idx_type], read_ret)
    def read_b_fn(bb, lds_off):
        frags = bb.read_multi_fragment_from_lds(
            lds_off, lds_read_tile_b, lds_swizzle, lds_read_sub_tile_b, bb.ds_read_b64
        )
        return [bb.to_any(d) for d, t in frags] + [t for d, t in frags]

    # -- Init accumulators --
    c_buf = b.memref_alloca(b.constant_index(n_accs), b.ax4_type)

    @b.foreach_tile(n_accs)
    def _(idx):
        b.memref_store(b.init_agprx4(b.constant_i32(0)), c_buf, idx)

    # -- Distribution --
    wg_m_idx, wg_n_idx = b.delinearize_index(b.linear_block_id(), (wg[DIM_M], wg[DIM_N]))
    wid = b.wave_id(wave_size=ws)
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (wpw[DIM_M], wpw[DIM_N]))
    m_dist_idx = b.linearize_index((wg_m_idx, wave_m_idx), (wg[DIM_M], wpw[DIM_M]))
    n_dist_idx = b.linearize_index((wg_n_idx, wave_n_idx), (wg[DIM_N], wpw[DIM_N]))

    # Cooperative load starts: wave_id -> (spatial_start, k_start) with OOB
    # clamping. Excess waves clamp to the last valid start and duplicate the
    # last tile. Same pattern as test_102.
    coop_a_m_start_raw = b.linearize_layout(wid, Layout((a_waves_m, a_waves_k), (coop_a_m, 0)))
    coop_a_k_start_raw = b.linearize_layout(wid, Layout((a_waves_m, a_waves_k), (0, coop_a_k)))
    coop_a_m_start = b.arith_minui(coop_a_m_start_raw, b.constant_index(max_a_m_start))
    coop_a_k_start = b.arith_minui(coop_a_k_start_raw, b.constant_index(max_a_k_start))
    coop_b_n_start_raw = b.linearize_layout(wid, Layout((b_waves_n, b_waves_k), (coop_b_n, 0)))
    coop_b_k_start_raw = b.linearize_layout(wid, Layout((b_waves_n, b_waves_k), (0, coop_b_k)))
    coop_b_n_start = b.arith_minui(coop_b_n_start_raw, b.constant_index(max_b_n_start))
    coop_b_k_start = b.arith_minui(coop_b_k_start_raw, b.constant_index(max_b_k_start))

    c0, c1 = b.constant_index(0), b.constant_index(1)
    s0, s1, s2 = [ir.AffineExpr.get_symbol(i) for i in range(3)]

    # -- K-loop: flat (no pipeline stages) --
    @b.loop(c0, b.constant_index(k_iters), c1)
    def _(k_iv):
        # -- LDS ALLOC --
        lds_a_h, lds_a = b.alloc_lds(lds_total_a)
        lds_b_h, lds_b = b.alloc_lds(lds_total_b)

        # -- G2S LOAD A (cooperative, 2D split across all waves) --
        a_wg_k_idx = b.linearize_index((wg_m_idx, k_iv), (wg[DIM_M], k_iters))
        a_wg_base = b.linearize_layout(a_wg_k_idx, WG_BASE_A)
        coop_a_off = b.linearize_layout(b.linearize_index((coop_a_k_start, coop_a_m_start), (k_t, twg_m)), TILE_COORD_A)
        a_wave_base = b.affine_apply(d0 + d1, [a_wg_base, coop_a_off])
        coop_a_lds_off = b.linearize_layout(
            b.linearize_index((coop_a_k_start, coop_a_m_start), (k_t, twg_m)), LDS_COORD_A
        )
        lds_a_wave = b.affine_apply(d0 + d1, [lds_a, coop_a_lds_off])

        @b.foreach_tile(n_coop_a, types=[(b.flat_write_tok, 1)])
        def g2s_toks_a(idx):
            tile_off = b.linearize_layout(idx, COOP_COORD_A)
            voff = b.affine_apply(s0 + s1 + s2, [], [a_wave_base, thread_off_a, tile_off])
            lds_off = b.affine_apply(d0 + d1, [lds_a_wave, b.linearize_layout(idx, COOP_LDS_A)])
            b.set_m0(m0, b.index_to_sgpr(lds_off))
            tok = b.g2s_buffer_load_dwordx4(m0, a_rsrc, soff_val, b.index_to_vgpr(voff))
            return [tok]

        @b.foreach_tile(n_coop_a)
        def _(i):
            b.wait_deps(b.memref_load(g2s_toks_a, i))

        # -- G2S LOAD B (cooperative, 2D split across all waves) --
        b_wg_k_idx = b.linearize_index((wg_n_idx, k_iv), (wg[DIM_N], k_iters))
        b_wg_base = b.linearize_layout(b_wg_k_idx, WG_BASE_B)
        coop_b_off = b.linearize_layout(b.linearize_index((coop_b_k_start, coop_b_n_start), (k_t, twg_n)), TILE_COORD_B)
        b_wave_base = b.affine_apply(d0 + d1, [b_wg_base, coop_b_off])
        coop_b_lds_off = b.linearize_layout(
            b.linearize_index((coop_b_k_start, coop_b_n_start), (k_t, twg_n)), LDS_COORD_B
        )
        lds_b_wave = b.affine_apply(d0 + d1, [lds_b, coop_b_lds_off])

        @b.foreach_tile(n_coop_b, types=[(b.flat_write_tok, 1)])
        def g2s_toks_b(idx):
            tile_off = b.linearize_layout(idx, COOP_COORD_B)
            voff = b.affine_apply(s0 + s1 + s2, [], [b_wave_base, thread_off_b, tile_off])
            lds_off = b.affine_apply(d0 + d1, [lds_b_wave, b.linearize_layout(idx, COOP_LDS_B)])
            b.set_m0(m0, b.index_to_sgpr(lds_off))
            tok = b.g2s_buffer_load_dwordx4(m0, b_rsrc, soff_val, b.index_to_vgpr(voff))
            return [tok]

        @b.foreach_tile(n_coop_b)
        def _(i):
            b.wait_deps(b.memref_load(g2s_toks_b, i))

        # -- BARRIER: wait for all G2S writes to complete --
        b.s_barrier()

        # -- LDS READ A --
        wave_m_off = b.linearize_layout(wid, WAVE_M_LDS_OFF)
        wave_lds_base_a = b.affine_apply(d0 + d1, [lds_a, wave_m_off])

        @b.foreach_tile(n_read_a, types=[(b.any_type, n_frags_per_tile), (b.lds_read_tok, n_frags_per_tile)])
        def read_a(idx):
            tile_off = b.linearize_layout(idx, WAVE_READ_COORD_A)
            off = b.affine_apply(d0 + d1, [wave_lds_base_a, tile_off])
            return b.call_helper(read_a_fn, [off], read_ret)

        frag_buf_a, rtok_buf_a = read_a
        b.dealloc_lds(lds_a_h)

        # -- LDS READ B --
        wave_n_off = b.linearize_layout(wid, WAVE_N_LDS_OFF)
        wave_lds_base_b = b.affine_apply(d0 + d1, [lds_b, wave_n_off])

        @b.foreach_tile(n_read_b, types=[(b.any_type, n_frags_per_tile), (b.lds_read_tok, n_frags_per_tile)])
        def read_b(idx):
            tile_off = b.linearize_layout(idx, WAVE_READ_COORD_B)
            off = b.affine_apply(d0 + d1, [wave_lds_base_b, tile_off])
            return b.call_helper(read_b_fn, [off], read_ret)

        frag_buf_b, rtok_buf_b = read_b
        b.dealloc_lds(lds_b_h)

        # -- COMPUTE: join 2 vx2 -> vx4 for CDNA4 MFMA --
        nf = n_frags_per_tile
        assert nf == 2, f"CDNA4 expects 2 vx2 frags per tile, got {nf}"

        @b.foreach_tile(k_t * m_t * n_t)
        def _(idx):
            kt, mt, nt = b.delinearize_index(idx, (k_t, m_t, n_t))
            acc_idx = b.linearize_index((mt, nt), (m_t, n_t))
            a_lo_i = b.linearize_index((kt, mt, b.constant_index(0)), (k_t, m_t, nf))
            a_hi_i = b.linearize_index((kt, mt, b.constant_index(1)), (k_t, m_t, nf))
            b_lo_i = b.linearize_index((kt, nt, b.constant_index(0)), (k_t, n_t, nf))
            b_hi_i = b.linearize_index((kt, nt, b.constant_index(1)), (k_t, n_t, nf))
            b.wait_deps(
                b.memref_load(rtok_buf_a, a_lo_i),
                b.memref_load(rtok_buf_a, a_hi_i),
                b.memref_load(rtok_buf_b, b_lo_i),
                b.memref_load(rtok_buf_b, b_hi_i),
            )
            acc = b.memref_load(c_buf, acc_idx)
            a_vx4 = b.join_vx2_to_vx4(
                b.from_any(b.memref_load(frag_buf_a, a_lo_i), b.vx2_type),
                b.from_any(b.memref_load(frag_buf_a, a_hi_i), b.vx2_type),
            )
            b_vx4 = b.join_vx2_to_vx4(
                b.from_any(b.memref_load(frag_buf_b, b_lo_i), b.vx2_type),
                b.from_any(b.memref_load(frag_buf_b, b_hi_i), b.vx2_type),
            )
            new_acc = b.mfma(MFMA_F16_CDNA4.opcode, acc, a_vx4, b_vx4)
            b.memref_store(new_acc, c_buf, acc_idx)

    # Store C tiles.
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
# Harness
# ---------------------------------------------------------------------------

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
    def label(self) -> str:
        return f"{super().label}_cdna4"

    @classmethod
    def from_label(cls, label: str) -> "Cdna4GemmInstance":
        suffix = "_cdna4"
        if not label.endswith(suffix):
            raise ValueError(f"Cannot parse CDNA4 label: {label}")
        base = WeakScaledMappedGemmInstance.from_label(label[: -len(suffix)])
        spec = GemmSpec.from_sizes(*base.gemm_size, mfma_shape=list(MFMA_F16_CDNA4.shape))
        mapping = dataclasses.replace(base.mapping, mcpu="gfx950")
        return cls(spec, mapping)


def _make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult):
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
        pipeline_strategy=0,
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
        pipeline = make_default_pass_pipeline(num_vgprs=256, num_agprs=256)
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
    """Execute a pre-compiled CDNA4 GEMM HSACO."""
    mcpu = getattr(cfg.mapping, "mcpu", "gfx950")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--wg", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--wpw", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--twg", type=int, nargs=3, default=[2, 2, 1])
    parser.add_argument("--k-mult", type=int, default=4)
    args = parser.parse_args()

    cfg = _make_instance(args.wg, args.wpw, args.twg, args.k_mult)
    gs = cfg.gemm_size
    print(
        f"Config: wg={'x'.join(map(str, args.wg))} wpw={'x'.join(map(str, args.wpw))} twg={'x'.join(map(str, args.twg))}"
    )
    print(f"  M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}, waves={cfg.mapping.num_waves}")

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        _, asm = compile_cdna4_gemm(cfg, f.name, print_ir_after_all=args.print_ir_after_all, print_asm=args.print_asm)
    if args.print_asm:
        print(asm)
