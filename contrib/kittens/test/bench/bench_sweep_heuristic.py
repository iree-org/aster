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
            ( 128,   128, 1024): "m128xn128xk1024_wg1x1x1_w2x2x1_twg8x8x2_pipestrat1_wgcu1_lcm1_um2_peel1_llsched1_hoistwait1_ldsw0_setprio1_rotc0_b_direct_ab_lt_flat",
        },
        "102": {
            ( 128,   128, 1024): "m128xn128xk1024_wg2x1x1_w2x2x1_twg4x8x1_pipestrat3_wgcu2_lcm1_um1_peel1_llsched1_hoistwait0_ldsw1_setprio0_rotc0_b_direct_b_lt_flat",
            (2432,  6144, 8192): "m2432xn6144xk8192_wg19x32x1_w1x4x1_twg8x12x1_pipestrat4_wgcu1_lcm0_um1_peel0_llsched1_hoistwait1_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (2432, 12288, 4096): "m2432xn12288xk4096_wg19x64x1_w1x4x1_twg8x12x1_pipestrat3_wgcu1_lcm1_um2_peel0_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (2432, 12288, 8192): "m2432xn12288xk8192_wg19x64x1_w1x4x1_twg8x12x1_pipestrat3_wgcu1_lcm1_um3_peel1_llsched1_hoistwait0_ldsw1_setprio1_rotc0_b_direct_b_lt_flat",
            (2432, 18432, 4096): "m2432xn18432xk4096_wg19x96x1_w1x4x1_twg8x12x1_pipestrat3_wgcu1_lcm0_um1_peel1_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (3648,  8192, 8192): "m3648xn8192xk8192_wg38x32x1_w1x4x1_twg6x16x1_pipestrat3_wgcu2_lcm0_um1_peel0_llsched1_hoistwait0_ldsw1_setprio1_rotc0_b_direct_b_lt_flat",
            (3648,  6144, 8192): "m3648xn6144xk8192_wg19x32x1_w2x2x1_twg12x12x1_pipestrat1_wgcu1_lcm1_um3_peel1_llsched0_hoistwait1_ldsw0_setprio1_rotc0_b_lds_lt_flat",
            (3648,  6144, 4096): "m3648xn6144xk4096_wg19x32x1_w2x2x1_twg12x12x1_pipestrat1_wgcu1_lcm1_um3_peel1_llsched0_hoistwait1_ldsw0_setprio1_rotc0_b_lds_lt_flat",
            (3648, 12288, 4096): "m3648xn12288xk4096_wg19x64x1_w2x2x1_twg12x12x1_pipestrat1_wgcu1_lcm1_um3_peel1_llsched0_hoistwait0_ldsw0_setprio1_rotc0_b_lds_lt_flat",
            (4096,  4096, 4096): "m4096xn4096xk4096_wg32x16x1_w1x4x1_twg8x16x1_pipestrat4_wgcu1_lcm1_um2_peel1_llsched1_hoistwait0_ldsw0_setprio0_rotc0_b_direct_b_lt_flat",
            (4864,  6144, 4096): "m4864xn6144xk4096_wg38x32x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um2_peel1_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864,  6144, 8192): "m4864xn6144xk8192_wg38x32x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um3_peel1_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864,  9216, 4096): "m4864xn9216xk4096_wg38x48x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um3_peel0_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864,  9216, 8192): "m4864xn9216xk8192_wg38x48x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um1_peel1_llsched1_hoistwait1_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864, 12288, 4096): "m4864xn12288xk4096_wg38x64x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm0_um1_peel0_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864, 15360, 4096): "m4864xn15360xk4096_wg38x80x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um1_peel1_llsched1_hoistwait0_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864, 18432, 4096): "m4864xn18432xk4096_wg38x96x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm0_um1_peel1_llsched1_hoistwait1_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
            (4864, 21504, 4096): "m4864xn21504xk4096_wg38x112x1_w1x4x1_twg8x12x1_pipestrat3_wgcu2_lcm1_um2_peel0_llsched1_hoistwait1_ldsw0_setprio1_rotc0_b_direct_b_lt_flat",
        },
        "103": {
            ( 128,   128, 1024): "m128xn128xk1024_wg2x1x1_w1x8x1_twg4x8x1_pipestrat1_wgcu1_lcm1_um3_peel0_llsched0_hoistwait1_ldsw1_setprio1_rotc0_b_direct_b_lt_flat",
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
    """Add --heuristic and --weak-scale-boost CLI args shared across benches."""
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Bias sampling toward promising configs",
    )
    parser.add_argument(
        "--weak-scale-boost",
        type=float,
        default=10.0,
        help="Priority boost for weak-scaled best-known configs (0 = disable seeding)",
    )


def generate_with_weak_scale(
    grid,
    mcpu: str,
    bench: str,
    target_m: int,
    target_n: int,
    target_k: int,
    args,
    *,
    sample_size: int,
    pins: dict | None = None,
    stratification_key=None,
):
    """Compose priority_fn + weak-scale seed and call ``grid.generate``.

    The composition is:
      - ``make_score_fn(mcpu, bench)`` when ``--heuristic`` is set
      - ``build_weak_scale_priority`` when ``--weak-scale-boost > 0`` and size is pinned
      - stratification is disabled automatically when any priority_fn is active
    """
    base_score_fn = make_score_fn(mcpu, bench) if getattr(args, "heuristic", False) else None
    extra_eligible, priority_fn = build_weak_scale_priority(
        mcpu,
        bench,
        target_m,
        target_n,
        target_k,
        base_score_fn,
        getattr(args, "weak_scale_boost", 0.0),
    )
    use_priority = priority_fn is not None
    return grid.generate(
        pins=pins or None,
        sample_size=sample_size,
        stratification_key=None if use_priority else stratification_key,
        priority_fn=priority_fn,
        extra_eligible=extra_eligible or None,
    )


def build_weak_scale_priority(
    mcpu: str,
    bench: str,
    target_M: int | None,
    target_N: int | None,
    target_K: int | None,
    base_score_fn: "callable | None",
    boost: float,
) -> tuple[list[dict], "callable | None"]:
    """Compose the weak-scale seeding for a sweep."""
    if boost <= 0 or target_M is None or target_N is None or target_K is None:
        return [], base_score_fn

    scaled = weak_scale_configs(mcpu, bench, target_M, target_N, target_K)
    if not scaled:
        return [], base_score_fn

    extra_eligible: list[dict] = []
    weak_scale_keys: set[tuple] = set()
    for inst, src_label in scaled:
        d = instance_to_axis_dict(inst)
        extra_eligible.append(d)
        weak_scale_keys.add(tuple(sorted(d.items())))
        print(f"  weak-scale seed: {inst.label}  (from {src_label[:40]}...)")
    print(f"  {len(extra_eligible)} weak-scaled configs added to eligible pool (boost={boost})")

    def priority_fn(d: dict) -> float:
        base = base_score_fn(d) if base_score_fn is not None else 0.0
        if tuple(sorted(d.items())) in weak_scale_keys:
            return base + boost
        return base

    return extra_eligible, priority_fn


def instance_to_axis_dict(inst: "WeakScaledMappedGemmInstance") -> dict:
    """Convert a WeakScaledMappedGemmInstance to an axis-level config dict."""
    from kittens.gemm_config import DIM_M, DIM_N, DIM_K

    m = inst.mapping
    gs = inst.gemm_size
    twg = m.num_tiles_per_workgroup
    waves = m.num_waves_per_workgroup
    wps_val = (waves[DIM_M] * waves[DIM_N] + 3) // 4
    return {
        "target_M": gs[DIM_M],
        "target_N": gs[DIM_N],
        "target_K": gs[DIM_K],
        "twg_m": twg[DIM_M],
        "twg_n": twg[DIM_N],
        "twg_k": m.num_tiles_per_wave[DIM_K],
        "waves_m": waves[DIM_M],
        "waves_n": waves[DIM_N],
        "occ": m.num_wg_per_cu * wps_val,
        "n_mult": 1,
        "k_factor": gs[DIM_K] // max(m.num_tiles_per_wave[DIM_K] * 32, 1),
        "ps": m.pipeline_strategy,
        "variant": (m.operand_path.value, m.load_type.value),
        "lcm_unroll": m.lcm_unroll,
        "unroll_mult": m.unroll_factor_multiplier,
        "epilogue_peeling": m.epilogue_peeling,
        "ll_sched": m.ll_sched,
        "hoist_wait": m.hoist_wait,
        "lds_at_write": m.lds_at_write,
        "set_mfma_priority": m.set_mfma_priority,
    }


def _factor_splits(factor: int, parts: int) -> list[tuple[int, ...]]:
    """All factorizations of `factor` into `parts` positive integer factors.

    Example: _factor_splits(4, 3) -> [(1,1,4), (1,2,2), (1,4,1), (2,1,2), (2,2,1), (4,1,1)]
    """
    if parts == 1:
        return [(factor,)]
    out: list[tuple[int, ...]] = []
    for d in range(1, factor + 1):
        if factor % d == 0:
            for rest in _factor_splits(factor // d, parts - 1):
                out.append((d,) + rest)
    return out


def weak_scale_configs(
    mcpu: str,
    bench: str,
    target_M: int,
    target_N: int,
    target_K: int,
) -> list[tuple["WeakScaledMappedGemmInstance", str]]:
    """Generate configs for (target_M, target_N, target_K) by weak-scaling best-known configs.

    Returns a list of (instance, source_label) pairs.
    """
    from kittens.gemm_config import GemmSpec, GemmMappingSpec, DIM_M, DIM_N, DIM_K
    from sweep_harness import WAVE_CONFIGS

    known = BEST_KNOWN.get(mcpu, {}).get(bench, {})
    if not known:
        return []

    results: list[tuple[WeakScaledMappedGemmInstance, str]] = []
    seen_labels: set[str] = set()
    wave_set = set(WAVE_CONFIGS)

    for (src_M, src_N, src_K), label in known.items():
        # Weak-scaling is strictly UP: source dimensions must divide target.
        if target_M % src_M != 0 or target_N % src_N != 0 or target_K % src_K != 0:
            continue
        if target_M < src_M or target_N < src_N or target_K < src_K:
            continue

        f_M = target_M // src_M
        f_N = target_N // src_N
        f_K = target_K // src_K

        src = WeakScaledMappedGemmInstance.from_label(label)
        m = src.mapping
        src_wg = m.num_workgroups_per_kernel
        src_waves = m.num_waves_per_workgroup
        src_tpw = m.num_tiles_per_wave

        # M: (wg_m_mult, waves_m_mult, tiles_per_wave_m_mult)
        # N: (wg_n_mult, waves_n_mult, tiles_per_wave_n_mult)
        # K: (k_iters_mult, twg_k_mult) -- k_iters is implicit (k_iters_mult
        #    is absorbed into the new target_K; we only track twg_k_mult).
        for m_wg, m_wv, m_tp in _factor_splits(f_M, 3):
            for n_wg, n_wv, n_tp in _factor_splits(f_N, 3):
                for _k_iters_mult, k_tp in _factor_splits(f_K, 2):
                    new_waves = [
                        src_waves[DIM_M] * m_wv,
                        src_waves[DIM_N] * n_wv,
                        src_waves[DIM_K],
                    ]
                    # Wave config must be in the valid set (MI300X limit is 16 waves/WG).
                    if (new_waves[DIM_M], new_waves[DIM_N]) not in wave_set:
                        continue

                    new_tpw = [
                        src_tpw[DIM_M] * m_tp,
                        src_tpw[DIM_N] * n_tp,
                        src_tpw[DIM_K] * k_tp,
                    ]
                    new_wg = [
                        src_wg[DIM_M] * m_wg,
                        src_wg[DIM_N] * n_wg,
                        src_wg[DIM_K],
                    ]

                    new_spec = GemmSpec.from_sizes(target_M, target_N, target_K)
                    new_mapping = GemmMappingSpec(
                        num_workgroups_per_kernel=new_wg,
                        num_waves_per_workgroup=new_waves,
                        num_tiles_per_wave=new_tpw,
                        pipeline_strategy=m.pipeline_strategy,
                        load_type=m.load_type,
                        operand_path=m.operand_path,
                        num_wg_per_cu=m.num_wg_per_cu,
                        lcm_unroll=m.lcm_unroll,
                        unroll_factor_multiplier=m.unroll_factor_multiplier,
                        epilogue_peeling=m.epilogue_peeling,
                        ll_sched=m.ll_sched,
                        hoist_wait=m.hoist_wait,
                        lds_at_write=m.lds_at_write,
                        set_mfma_priority=m.set_mfma_priority,
                    )
                    try:
                        inst = WeakScaledMappedGemmInstance(new_spec, new_mapping)
                    except AssertionError:
                        continue
                    if inst.label in seen_labels:
                        continue
                    seen_labels.add(inst.label)
                    results.append((inst, label))

    return results


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
