# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TierSpec dataclass + tier-mechanism helpers (policy lives in benches)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench_search import SweepGrid


def make_constraints(grid) -> tuple[str, ...]:
    """Return the named filters registered on `grid`."""
    return tuple(f.name for f in grid._filters if f.name is not None)


@dataclass(frozen=True)
class TierSpec:
    """Declarative description of one tier.

    Axis-value resolution (per axis, later wins):
      1. Tier-N>1: per-winner expansion (neighbor_radius + free_axes).
      2. axis_grid (multi-value): explicit enumeration.
      3. fixed_axes (single-value): pin.
      4. ambient_pins (driver target_M/N/K): always wins.

    `discriminator` is an axis name or tuple of names used as the
    stratum key for sampling. None = uniform.

    `per_stratum_diversity`: keep the best one per discriminator stratum
    (then take top 10% by TF), instead of overall top 10%. Used for
    geometric diversity.

    `anchor_axes`: per tier-(N-1) winner, the driver synthesizes
    (axis_grid x anchor_axes) configs and protects them from sampling.
    Other axis_grid values are still swept normally.

    `constraints`: named filters from the grid that this tier keeps;
    others are dropped. Anonymous filters always apply.
    """

    tier_idx: int
    max_configs: int
    random_seed: int
    constraints: tuple[str, ...]
    axis_grid: dict[str, list[Any]] = field(default_factory=dict)
    fixed_axes: dict[str, Any] = field(default_factory=dict)
    free_axes: list[str] = field(default_factory=list)
    neighbor_radius: dict[str, int] = field(default_factory=dict)
    discriminator: str | tuple[str, ...] | None = None
    per_stratum_diversity: bool = False
    anchor_axes: dict[str, Any] = field(default_factory=dict)


def _ordinal_neighbors(value: Any, value_list: list[Any], radius: int) -> list[Any]:
    """Values within `radius` index steps of `value` (clamped at boundaries)."""
    if value not in value_list:
        return [value]
    i = value_list.index(value)
    lo = max(0, i - radius)
    hi = min(len(value_list) - 1, i + radius)
    return value_list[lo : hi + 1]


def _tier_axis_overrides(
    ref_axis_vals: dict[str, list[Any]],
    tier: TierSpec,
    prev_winners: list[dict[str, Any]],
    ambient_pins: dict[str, Any],
) -> dict[str, list[Any]]:
    """{axis_name -> values} for the tier's per-tier grid (see TierSpec for resolution order)."""
    overrides: dict[str, list[Any]] = {}

    if prev_winners:
        from collections import defaultdict

        union: dict[str, set] = defaultdict(set)
        for winner in prev_winners:
            base = {k: v for k, v in winner.items() if k != "_tflops"}
            for k, v in base.items():
                if k in ref_axis_vals:
                    union[k].add(v)
            for axis_name, radius in tier.neighbor_radius.items():
                if axis_name in base and axis_name in ref_axis_vals:
                    for v in _ordinal_neighbors(base[axis_name], list(ref_axis_vals[axis_name]), radius):
                        union[axis_name].add(v)
            for axis_name in tier.free_axes:
                if axis_name in ref_axis_vals:
                    for v in ref_axis_vals[axis_name]:
                        union[axis_name].add(v)

        for k, vs in union.items():
            overrides[k] = sorted(vs, key=str)

    for k, vals in tier.axis_grid.items():
        overrides[k] = list(vals)
    for k, v in tier.fixed_axes.items():
        overrides[k] = [v]
    for k, v in ambient_pins.items():
        overrides[k] = [v]

    return overrides


def apply_tier_overrides(
    grid: "SweepGrid",
    tier: TierSpec,
    prev_winners: list[dict[str, Any]],
    ambient_pins: dict[str, Any] | None = None,
    *,
    universe: dict[str, list[Any]] | None = None,
) -> None:
    """Mutate `grid` in place to enforce `tier`'s policy.

    `universe` is the value pool used by tier-N>1 ordinal-neighbor
    expansion (typically tier-1's effective axis values). When None,
    falls back to the grid's registered values.
    """
    if universe is None:
        universe = grid.axis_values()
    overrides = _tier_axis_overrides(universe, tier, prev_winners, ambient_pins or {})
    grid.restrict_axes(overrides)
    grid.retain_filters(set(tier.constraints))
    empty_axes = [a.name for a in grid._axes if not a.values]
    if empty_axes:
        raise ValueError(
            f"Tier {tier.tier_idx} left sweep axes with no values {empty_axes!r}; "
            "add them to axis_grid, fixed_axes, or apply_bench_scheduling_defaults()"
        )
