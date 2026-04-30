# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep heuristic: best-known configs + rules for ordering sweep candidates."""

from __future__ import annotations

import os
import sys

# Path setup -- same as all bench scripts in this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from kittens.gemm_config import WeakScaledMappedGemmInstance


# ---------------------------------------------------------------------------
# Pin dict shorthand -- keeps data tables on one line per entry.
# ---------------------------------------------------------------------------


def _p(twg_m, twg_n, waves_m, waves_n, ps=None, db=True, **kw):
    """Build a pin dict.

    Short arg names for compact table rows.
    """
    d = {"tiles-per-wg-m": twg_m, "tiles-per-wg-n": twg_n, "waves-per-wg-m": waves_m, "waves-per-wg-n": waves_n}
    if ps is not None:
        d["pipeline-strategy"] = ps
    if db:
        d["direct-b"] = ""
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# Best-known configs: mcpu -> bench -> (M, N, K) -> serde label
#
# Benches are split by target GPU arch: gfx942 hosts bench_perf_001, _102, _103;
# gfx950 hosts bench_perf_102_..._cdna4 (and future CDNA4-only benches).
# A given bench key only appears under the mcpu it actually targets.
#
# Add new static entries as sweeps discover better configs.
# The label is the canonical serde format, one line per entry.
# Keep sorted by (M, N, K) within each bench.
# ---------------------------------------------------------------------------

# fmt: off
BEST_KNOWN: dict[str, dict[str, dict[tuple[int, int, int], str]]] = {
    "gfx942": {
        "001": {
        },
        "102": {
        },
        "103": {
        },
    },
    "gfx950": {
        "102_cdna4": {
        },
    },
}
# fmt: on

# ---------------------------------------------------------------------------
# Heuristic rules: mcpu -> bench -> ordered list of partial pin dicts.
# Higher-ranked configs are tried first.
# ---------------------------------------------------------------------------

# TODO: atm twg_m, twg_n, waves_m, waves_n require divisibility. Relax this in the future.
HEURISTIC_RULES: dict[str, dict[str, list[dict]]] = {
    "gfx942": {
        "102": [
            _p(8, 12, 1, 4, ps=3),
            _p(8, 12, 1, 4, ps=4),
            _p(8, 12, 1, 4, ps=1),
            _p(12, 12, 2, 2, ps=1, db=False),
            _p(8, 14, 1, 4),
            _p(8, 10, 1, 4),
            _p(8, 16, 1, 4),
            _p(8, 16, 1, 8),
            _p(6, 16, 1, 4),
            _p(8, 12, 2, 2),
            _p(10, 12, 1, 4),
        ],
    },
    "gfx950": {
        "102_cdna4": [],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def best_known(mcpu: str, bench: str, M: int, N: int, K: int) -> str | None:
    """Return the best known config label for (mcpu, bench, M, N, K), or None."""
    return BEST_KNOWN.get(mcpu, {}).get(bench, {}).get((M, N, K))


def add_heuristic_cli_args(parser) -> None:
    """Add --heuristic CLI arg shared across benches."""
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Bias sampling toward promising configs",
    )


_PREFERRED_FEATURES: dict[str, dict[str, dict]] = {
    "gfx942": {
        "twg_n": {12: 0.47, 16: 0.30, 24: 0.25, 14: 0.20, 10: 0.15, 20: 0.10, 8: 0.05},
        "twg_m": {8: 0.10, 12: 0.08, 6: 0.03, 10: 0.03},
        "variant": {"direct_b": 0.07},
        "ps": {3: 0.10, 4: 0.08, 1: 0.05, 2: 0.05},
        "waves_m": {1: 0.12, 2: 0.10},
        "waves_n": {4: 0.04, 8: 0.03, 2: 0.02},
        "occ": {2: 0.05, 1: 0.04, 3: 0.01},
        "ll_sched": {True: 0.03},
    },
    "gfx950": {},
}


def make_score_fn(mcpu: str, bench: str) -> callable:
    """Return a scoring function for config dicts (axis-level keys).

    Higher score = more promising config. Used as ``priority_fn`` in
    ``SweepGrid.generate()`` for weighted sampling.
    """
    rules = HEURISTIC_RULES.get(mcpu, {}).get(bench, [])
    axis_rules = [(to_axis_pins(r), 1.0 / (1 + i)) for i, r in enumerate(rules)]
    preferred = _PREFERRED_FEATURES.get(mcpu, {})

    def score(d: dict) -> float:
        s = 0.0
        # Per-feature bonus from preferred values.
        for feat, val_scores in preferred.items():
            s += val_scores.get(d.get(feat), 0.0)
        # Exact rule match bonus (stacks with feature bonuses).
        for axis_rule, weight in axis_rules:
            if all(d.get(k) == v for k, v in axis_rule.items()):
                s += weight
        return s

    return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Single table driving both label->pins and pins->axis conversions.
#
# Each entry: (pin_key, axis_name, accessor, emit_val, axis_val)
#   accessor(mapping) -> current value from GemmMappingSpec
#   emit_val: the accessor result that triggers emitting this pin.
#       None means "always emit" (required pins like tile counts).
#   axis_val: value to set on the sweep axis when this pin is present.
#       None means "use the pin's value directly".
#
# Example: ("ll-sched", "ll_sched", lambda m: m.ll_sched, True, True)
#   Emitted when ll_sched==True. Sets axis ll_sched=True in to_axis_pins.
# Example: ("no-lcm-unroll", "lcm_unroll", lambda m: m.lcm_unroll, False, False)
#   Emitted when lcm_unroll==False. Sets axis lcm_unroll=False.
# fmt: off
_PIN_SPEC = [
    ("tiles-per-wg-m",           "twg_m",                    lambda m: m.num_tiles_per_workgroup[0],  None,  None),
    ("tiles-per-wg-n",           "twg_n",                    lambda m: m.num_tiles_per_workgroup[1],  None,  None),
    ("waves-per-wg-m",           "waves_m",                  lambda m: m.num_waves_per_workgroup[0],  None,  None),
    ("waves-per-wg-n",           "waves_n",                  lambda m: m.num_waves_per_workgroup[1],  None,  None),
    ("pipeline-strategy",        "ps",                       lambda m: m.pipeline_strategy,           None,  None),
    ("desired-simd-occupancy",   "occ_pin",                  lambda m: m.num_wg_per_cu * ((m.num_waves_per_workgroup[0] * m.num_waves_per_workgroup[1] + 3) // 4), None, None),
    ("direct-b",                 "variant",                  lambda m: m.operand_path.value in ("direct_b", "direct_ab"), True, "direct_b"),
    ("unroll-factor-multiplier", "unroll_factor_multiplier", lambda m: m.unroll_factor_multiplier,    None,  None),
    ("no-lcm-unroll",            "lcm_unroll",               lambda m: m.lcm_unroll,                  False, False),
    ("no-epilogue-peeling",      "epilogue_peeling",         lambda m: m.epilogue_peeling,            False, False),
    ("ll-sched",                 "ll_sched",                 lambda m: m.ll_sched,                    True,  True),
    ("hoist-wait",               "hoist_wait",               lambda m: m.hoist_wait,                  True,  True),
    ("lds-at-write",             "lds_at_write",             lambda m: m.lds_at_write,                True,  True),
    ("no-set-mfma-priority",     "set_mfma_priority",        lambda m: m.set_mfma_priority,           False, False),
]
# fmt: on


def to_axis_pins(heuristic_pins: dict) -> dict:
    """Convert a heuristic pin dict (CLI-style keys) to axis-level pins."""
    lookup = {pin_key: (axis, axis_val) for pin_key, axis, _, _, axis_val in _PIN_SPEC}
    out = {}
    for key, val in heuristic_pins.items():
        if key not in lookup:
            continue
        axis, fixed = lookup[key]
        out[axis] = fixed if fixed is not None else val
    return out


def _label_to_pins(label: str) -> dict:
    """Extract sweep-compatible pins from a serde label."""
    cfg = WeakScaledMappedGemmInstance.from_label(label)
    m = cfg.mapping
    pins: dict = {}
    for pin_key, _, accessor, emit_val, _ in _PIN_SPEC:
        val = accessor(m)
        if emit_val is None or val == emit_val:
            pins[pin_key] = val if not isinstance(val, bool) else ""
    return pins
