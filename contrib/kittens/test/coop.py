"""Cooperative-load helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aster import ir
from aster.layout import Layout, Symbol


def coop_2d_split(num_tiles, num_waves, kt):
    """Pick (waves_s, waves_k, coop_s, coop_k) minimizing clamping waste."""
    best = None
    for ws in range(1, num_waves + 1):
        if num_waves % ws != 0:
            continue
        wk = num_waves // ws
        cs = math.ceil(num_tiles / ws)
        ck = math.ceil(kt / wk)
        wasted = ws * wk * cs * ck - num_tiles * kt
        over_excess = max(0, ws - num_tiles) + max(0, wk - kt)
        key = (wasted, over_excess, -ws)
        if best is None or key < best[0]:
            best = (key, ws, wk, cs, ck)
    _, waves_s, waves_k, coop_s, coop_k = best
    return waves_s, waves_k, coop_s, coop_k


@dataclass(frozen=True, slots=True)
class CoopLoadPlan:
    """Per-operand cooperative-load output of make_coop_load_plan.

    Five fields consumed at the call site:
      - global_layout / lds_layout: rank-2 per-wave coop-iter Layouts
        (sizes (coop_s, coop_k)), used as Tensor layouts for the global
        load and LDS write respectively.
      - global_wave_off / lds_wave_off: clamped wave-start byte offsets
        added to the Tensor offset before transfer_tiles.
      - unroll_axes: (spatial_axis, k_axis) tuple for transfer_tiles.
    """

    global_layout: Layout
    global_wave_off: ir.Value
    lds_layout: Layout
    lds_wave_off: ir.Value
    unroll_axes: tuple


def make_coop_load_plan(
    b,
    wid,
    *,
    num_waves: int,
    wg_tile_global: Layout,
    wg_tile_lds: Layout,
    spatial_axis: Symbol,
    k_axis: Symbol,
) -> CoopLoadPlan:
    """Build a CoopLoadPlan from per-tile WG-level Layouts."""
    num_tiles, kt = wg_tile_global.sizes
    s_stride_g, k_stride_g = wg_tile_global.strides
    s_stride_l, k_stride_l = wg_tile_lds.strides
    assert wg_tile_lds.sizes == (num_tiles, kt), (
        f"wg_tile_lds sizes {wg_tile_lds.sizes} != wg_tile_global {wg_tile_global.sizes}"
    )

    waves_s, waves_k, coop_s, coop_k = coop_2d_split(num_tiles, num_waves, kt)
    max_s_start = max(0, num_tiles - coop_s)
    max_k_start = max(0, kt - coop_k)

    per_wave_global = Layout((coop_s, coop_k), (s_stride_g, k_stride_g), axes=(spatial_axis, k_axis))
    per_wave_lds = Layout((coop_s, coop_k), (s_stride_l, k_stride_l), axes=(spatial_axis, k_axis))

    wave_s_idx, wave_k_idx = b.delinearize_index(wid, (waves_s, waves_k))
    d0 = ir.AffineExpr.get_dim(0)

    # Clamp in tile units, then convert to byte offsets via layout_apply.
    # Multiplying by stride before arith_minui puts large byte literals (e.g.
    # 40960) into the v_cndmask immediate slot and trips AMDGCN.
    s_start_tiles = b.arith_minui(b.affine_apply(d0 * coop_s, [wave_s_idx]), b.constant_index(max_s_start))
    k_start_tiles = b.arith_minui(b.affine_apply(d0 * coop_k, [wave_k_idx]), b.constant_index(max_k_start))
    wg_global_full = Layout((num_tiles, kt), (s_stride_g, k_stride_g), axes=(spatial_axis, k_axis))
    wg_lds_full = Layout((num_tiles, kt), (s_stride_l, k_stride_l), axes=(spatial_axis, k_axis))
    global_wave_off = b.layout_apply((s_start_tiles, k_start_tiles), wg_global_full)
    lds_wave_off = b.layout_apply((s_start_tiles, k_start_tiles), wg_lds_full)

    return CoopLoadPlan(
        global_layout=per_wave_global,
        global_wave_off=global_wave_off,
        lds_layout=per_wave_lds,
        lds_wave_off=lds_wave_off,
        unroll_axes=(spatial_axis, k_axis),
    )
