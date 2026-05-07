# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Minimal tests for the tier-mechanism in `bench_tier_schedule`.

Scope is intentionally narrow:
  - `_ordinal_neighbors` (clamping, value-not-in-list)
  - `_allowed_variants` (close / far / threshold)
  - `_tier_axis_overrides` (tier-1 enumerates axis_grid + fixed_axes;
    tier-N>1 unions per-winner ordinal+boolean expansions; ambient pins
    win; variant gating fires at tier-2)

Bench-specific schedules (the `TIER_SCHEDULE` constants in each
`bench_perf_xxx.py`) are policy, not mechanism, and are not tested here.
The `apply_tier_overrides` + `grid_factory` integration with a real
`SweepGrid` (and the `run_tier_mode` driver) is exercised end-to-end by
the bench's hotaisle runs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from bench_tier_schedule import (  # noqa: E402
    VARIANT_GAP_PCT,
    TierSpec,
    _allowed_variants,
    _ordinal_neighbors,
    _tier_axis_overrides,
)


def _ref_axis_vals() -> dict[str, list]:
    """Tiny reference universe: one ordinal axis, one boolean, one categorical."""
    return {"a": [1, 2, 3, 4], "b": [True, False], "c": ["x", "y"]}


def _make_tier(tier_idx: int, **kw) -> TierSpec:
    """TierSpec helper that fills in the required scalar fields.

    The test fake grid has no named filters, so `constraints=()` is
    fine.
    """
    defaults = dict(top_k_to_keep=2, max_configs=100, random_seed=42, constraints=())
    defaults.update(kw)
    return TierSpec(tier_idx=tier_idx, **defaults)


def _generate_tier_configs(tier, prev_winners, ambient_pins=None):
    """Test helper: enumerate the per-tier axis-value overrides as full configs.

    Mirrors what `SweepGrid.generate` would enumerate for a tier_grid built
    from `_ref_axis_vals()`, before any filter pruning. Used by the tests
    below to exercise `_tier_axis_overrides` without spinning up a SweepGrid.
    """
    import itertools as _it

    overrides = _tier_axis_overrides(_ref_axis_vals(), tier, prev_winners, ambient_pins or {})
    keys = list(overrides.keys())
    return [dict(zip(keys, combo)) for combo in _it.product(*[overrides[k] for k in keys])]


# ---------------------------------------------------------------------------
# _ordinal_neighbors
# ---------------------------------------------------------------------------


def test_ordinal_neighbors_basic():
    assert _ordinal_neighbors(2, [1, 2, 3, 4], 1) == [1, 2, 3]


def test_ordinal_neighbors_clamps_at_boundaries():
    assert _ordinal_neighbors(1, [1, 2, 3, 4], 1) == [1, 2]
    assert _ordinal_neighbors(4, [1, 2, 3, 4], 1) == [3, 4]


def test_ordinal_neighbors_value_not_in_list_returns_singleton():
    assert _ordinal_neighbors(99, [1, 2, 3, 4], 1) == [99]


# ---------------------------------------------------------------------------
# _allowed_variants (variant gating)
# ---------------------------------------------------------------------------


def test_variant_gating_keeps_both_when_within_threshold():
    winners = [
        {"variant": "x", "_tflops": 100.0},
        {"variant": "y", "_tflops": 100.0 * (1 - VARIANT_GAP_PCT / 100 / 2)},
    ]
    assert _allowed_variants(winners) == {"x", "y"}


def test_variant_gating_pins_when_outside_threshold():
    winners = [
        {"variant": "x", "_tflops": 100.0},
        {"variant": "y", "_tflops": 100.0 * (1 - 2 * VARIANT_GAP_PCT / 100)},
    ]
    assert _allowed_variants(winners) == {"x"}


def test_variant_gating_exactly_at_threshold_keeps_both():
    winners = [
        {"variant": "x", "_tflops": 100.0},
        {"variant": "y", "_tflops": 100.0 * (1 - VARIANT_GAP_PCT / 100)},
    ]
    assert len(_allowed_variants(winners)) == 2


# ---------------------------------------------------------------------------
# _tier_axis_overrides + tier-1 / tier-N axis-value computation
# ---------------------------------------------------------------------------


def test_tier1_axis_overrides_combines_axis_grid_with_fixed_axes():
    tier = _make_tier(1, axis_grid={"a": [1, 2]}, fixed_axes={"b": True, "c": "x"})
    configs = _generate_tier_configs(tier, [], ambient_pins={})
    assert len(configs) == 2
    assert all(c["b"] is True and c["c"] == "x" for c in configs)
    assert {c["a"] for c in configs} == {1, 2}


def test_tier_n_axis_overrides_expands_ordinal_and_flips_free_axes():
    tier_n = _make_tier(3, neighbor_radius={"a": 1}, free_axes=["b"])
    winner = {"a": 2, "b": True, "c": "x", "_tflops": 100.0}
    configs = _generate_tier_configs(tier_n, [winner])
    # `_tflops` annotation must not survive into the output cfgs.
    assert all("_tflops" not in c for c in configs)
    # Ordinal radius 1 around a=2 in [1,2,3,4] -> {1, 2, 3}.
    assert {c["a"] for c in configs} == {1, 2, 3}
    # Boolean axis "b" in free_axes -> both True and False surface.
    assert {c["b"] for c in configs} == {True, False}


def test_tier_n_axis_overrides_keeps_winner_value():
    tier_n = _make_tier(3, neighbor_radius={"a": 1}, free_axes=[])
    winner = {"a": 2, "b": True, "c": "x", "_tflops": 100.0}
    configs = _generate_tier_configs(tier_n, [winner])
    # Winner's value for the ordinal axis is included in the neighborhood.
    assert 2 in {c["a"] for c in configs}


def test_tier2_applies_variant_gating():
    """Tier-2 with one dominating variant should drop the loser."""
    tier2 = _make_tier(2, neighbor_radius={"a": 1})
    winners = [
        {"a": 2, "variant": "x", "_tflops": 100.0},
        {"a": 2, "variant": "y", "_tflops": 50.0},  # 50% gap >> VARIANT_GAP_PCT
    ]
    # Tier-2 needs "variant" in ref_axis_vals so the override mechanism sees it.
    import bench_tier_schedule as bts

    overrides = bts._tier_axis_overrides(
        {**_ref_axis_vals(), "variant": ["x", "y"]},
        tier2,
        winners,
        ambient_pins={},
    )
    assert overrides["variant"] == ["x"]


def test_ambient_pins_override_axis_grid():
    tier = _make_tier(1, axis_grid={"a": [1, 2, 3]}, fixed_axes={})
    overrides = _tier_axis_overrides(_ref_axis_vals(), tier, [], ambient_pins={"a": 2})
    # ambient_pins always pin to a single value, regardless of axis_grid.
    assert overrides["a"] == [2]


# ---------------------------------------------------------------------------
# TierSpec construction sanity (required-field round-trip).
# ---------------------------------------------------------------------------


def test_tier_spec_required_fields_round_trip():
    t = TierSpec(tier_idx=1, top_k_to_keep=4, max_configs=200, random_seed=17, constraints=("foo",))
    assert t.tier_idx == 1
    assert t.top_k_to_keep == 4
    assert t.max_configs == 200
    assert t.random_seed == 17
    assert t.constraints == ("foo",)


def test_tier_spec_requires_constraints():
    """Constructing a TierSpec without `constraints=` must fail loudly."""
    import pytest

    with pytest.raises(TypeError):
        TierSpec(tier_idx=1, top_k_to_keep=4, max_configs=200, random_seed=17)
