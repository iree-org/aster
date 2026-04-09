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
# Best-known configs: (bench, M, N, K) -> serde label
#
# Add new static entries as sweeps discover better configs.
# The label is the canonical serde format, one line per entry.
# Keep sorted by (M, N, K) within each bench.
# ---------------------------------------------------------------------------

# fmt: off
BEST_KNOWN: dict[str, dict[tuple[int, int, int], str]] = {
    "102": {
        (2432,  6144, 8192): "m2432xn6144xk8192_wg19x32x1_w1x4x1_twg8x12x1_pipestrat4_nolcm_nopeel_llsched_hoistwait_direct_b_flat",
        (2432, 12288, 4096): "m2432xn12288xk4096_wg19x64x1_w1x4x1_twg8x12x1_pipestrat3_um2_nopeel_llsched_direct_b_flat",
        (2432, 12288, 8192): "m2432xn12288xk8192_wg19x64x1_w1x4x1_twg8x12x1_pipestrat3_um3_llsched_ldsw_direct_b_flat",
        (2432, 18432, 4096): "m2432xn18432xk4096_wg19x96x1_w1x4x1_twg8x12x1_pipestrat3_nolcm_llsched_direct_b_flat",
        (3648,  8192, 8192): "m3648xn8192xk8192_wg38x32x1_w1x4x1_twg6x16x1_pipestrat3_wgcu2_nolcm_nopeel_llsched_ldsw_direct_b_flat",
        (3648,  6144, 8192): "m3648xn6144xk8192_wg19x32x1_w2x2x1_twg12x12x1_pipestrat1_um3_hoistwait_flat",
        (3648,  6144, 4096): "m3648xn6144xk4096_wg19x32x1_w2x2x1_twg12x12x1_pipestrat1_um3_hoistwait_flat",
        (3648, 12288, 4096): "m3648xn12288xk4096_wg19x64x1_w2x2x1_twg12x12x1_pipestrat1_um3_flat",
        (4096,  4096, 4096): "m4096xn4096xk4096_wg32x16x1_w1x4x1_twg8x16x1_pipestrat4_um2_llsched_nosetprio_direct_b_flat",
        (4864,  6144, 4096): "m4864xn6144xk4096_wg38x32x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_um2_llsched_direct_b_flat",
        (4864,  6144, 8192): "m4864xn6144xk8192_wg38x32x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_um3_llsched_direct_b_flat",
        (4864,  9216, 4096): "m4864xn9216xk4096_wg38x48x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_um3_nopeel_llsched_direct_b_flat",
        (4864,  9216, 8192): "m4864xn9216xk8192_wg38x48x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_llsched_hoistwait_direct_b_flat",
        (4864, 12288, 4096): "m4864xn12288xk4096_wg38x64x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_nolcm_nopeel_llsched_direct_b_flat",
        (4864, 15360, 4096): "m4864xn15360xk4096_wg38x80x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_llsched_direct_b_flat",
        (4864, 18432, 4096): "m4864xn18432xk4096_wg38x96x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_nolcm_llsched_hoistwait_direct_b_flat",
        (4864, 21504, 4096): "m4864xn21504xk4096_wg38x112x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_um2_nopeel_llsched_hoistwait_direct_b_flat",
    },
}
# fmt: on

# ---------------------------------------------------------------------------
# Heuristic rules: higher-ranked configs are tried first.
# ---------------------------------------------------------------------------

# TODO: atm twg_m, twg_n, waves_m, waves_n require divisibility. Relax this in the future.
HEURISTIC_RULES_MI300X: dict[str, list[dict]] = {
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
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def best_known(bench: str, M: int, N: int, K: int) -> str | None:
    """Return the best known config label for (bench, M, N, K), or None."""
    return BEST_KNOWN.get(bench, {}).get((M, N, K))


def add_heuristic_cli_args(parser) -> None:
    """Add --heuristic CLI arg shared across benches."""
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Bias sampling toward promising configs",
    )


def make_score_fn(bench: str) -> callable:
    """Return a scoring function for config dicts (axis-level keys).

    Higher score = more promising config. Used as ``priority_fn`` in
    ``SweepGrid.generate()`` for weighted sampling.
    """
    rules = HEURISTIC_RULES_MI300X.get(bench, [])
    axis_rules = [(to_axis_pins(r), 1.0 / (1 + i)) for i, r in enumerate(rules)]

    _PREFERRED_MI300X = {
        "twg_n": {12: 0.47, 16: 0.30, 24: 0.25, 14: 0.20, 10: 0.15, 20: 0.10, 8: 0.05},
        "twg_m": {8: 0.10, 12: 0.08, 6: 0.03, 10: 0.03},
        "variant": {"direct_b": 0.07},
        "ps": {3: 0.10, 4: 0.08, 1: 0.05, 2: 0.05},
        "waves_m": {1: 0.12, 2: 0.10},
        "waves_n": {4: 0.04, 8: 0.03, 2: 0.02},
        "occ": {2: 0.05, 1: 0.04, 3: 0.01},
        "ll_sched": {True: 0.03},
    }
    # fmt: on

    def score(d: dict) -> float:
        s = 0.0
        # Per-feature bonus from preferred values.
        for feat, val_scores in _PREFERRED_MI300X.items():
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
    ("tiles-per-wg-m",          "twg_m",             lambda m: m.num_tiles_per_workgroup[0],  None,  None),
    ("tiles-per-wg-n",          "twg_n",             lambda m: m.num_tiles_per_workgroup[1],  None,  None),
    ("waves-per-wg-m",          "waves_m",           lambda m: m.num_waves_per_workgroup[0],  None,  None),
    ("waves-per-wg-n",          "waves_n",           lambda m: m.num_waves_per_workgroup[1],  None,  None),
    ("pipeline-strategy",       "ps",                lambda m: m.pipeline_strategy,           None,  None),
    ("desired-simd-occupancy",  "occ_pin",           lambda m: m.num_wg_per_cu * ((m.num_waves_per_workgroup[0] * m.num_waves_per_workgroup[1] + 3) // 4), None, None),
    ("direct-b",                "variant",           lambda m: m.operand_path.value in ("direct_b", "direct_ab"), True, "direct_b"),
    ("unroll-factor-multiplier","unroll_mult",       lambda m: m.unroll_factor_multiplier,    None,  None),
    ("no-lcm-unroll",           "lcm_unroll",        lambda m: m.lcm_unroll,                  False, False),
    ("no-epilogue-peeling",     "epilogue_peeling",  lambda m: m.epilogue_peeling,            False, False),
    ("ll-sched",                "ll_sched",          lambda m: m.ll_sched,                    True,  True),
    ("hoist-wait",              "hoist_wait",         lambda m: m.hoist_wait,                  True,  True),
    ("lds-at-write",            "lds_at_write",      lambda m: m.lds_at_write,                True,  True),
    ("no-set-mfma-priority",    "set_mfma_priority", lambda m: m.set_mfma_priority,           False, False),
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
