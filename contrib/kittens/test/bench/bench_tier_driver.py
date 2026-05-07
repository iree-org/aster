# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tiered successive-halving sweep driver."""

from __future__ import annotations

import itertools
import os
import random
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(__file__))

from bench_harness import bench_perf_sweep_pipelined  # noqa: E402
from bench_search import verify_top_configs  # noqa: E402
from bench_tier_schedule import TierSpec, apply_tier_overrides  # noqa: E302, E402


def run_tier_mode(
    args,
    *,
    target_m: int,
    target_n: int,
    target_k: int,
    grid_factory: Callable,
    compile_fn: Callable,
    repro_cmd_fn: Callable,
    post_compile_filter: Callable | None,
    bench_label: str,
    tier_schedule: list[TierSpec],
) -> str | None:
    """Walk `tier_schedule`. Each tier samples, runs, keeps top-K to seed the next.

    Returns the txt report path, or None if no tier produced measurable
    results.
    """
    schedule = tier_schedule
    ambient_pins = {"target_M": target_m, "target_N": target_n, "target_K": target_k}
    prev_winners: list[dict] = []
    all_records: list[dict] = []
    all_results: list[tuple] = []
    all_hsaco_paths: dict = {}
    out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or "/tmp"
    out_path = os.path.join(out_dir, f"tier_results_{bench_label}_{int(time.time())}.txt")

    # Tier-1's effective axis values seed ordinal-neighbor expansion in later tiers.
    tier1_universe: dict = {}

    for tier in schedule:
        cap = tier.max_configs
        if tier.random_seed is not None:
            random.seed(tier.random_seed)
        print(
            f"\n=== Tier {tier.tier_idx}: cap={cap}, top_k={tier.top_k_to_keep}, "
            f"discriminator={tier.discriminator!r}, seed={tier.random_seed!r} ==="
        )

        tier_grid = grid_factory()
        apply_tier_overrides(tier_grid, tier, prev_winners, ambient_pins=ambient_pins, universe=tier1_universe)
        if not prev_winners:
            tier1_universe = tier_grid.axis_values()

        if isinstance(tier.discriminator, tuple):
            stratification_key = lambda d, ks=tier.discriminator: tuple(d[k] for k in ks)  # noqa: E731
        elif tier.discriminator:
            stratification_key = lambda d, k=tier.discriminator: d[k]  # noqa: E731
        else:
            stratification_key = None

        anchor_extras = _build_anchor_extras(tier, prev_winners, ambient_pins)
        instances, cfgs, total = tier_grid.generate(
            sample_size=cap,
            stratification_key=stratification_key,
            extra_eligible=anchor_extras or None,
        )
        print(f"Tier {tier.tier_idx} population: generated={total}, clipped={len(instances)} (cap={cap})")

        if not instances:
            print(f"Tier {tier.tier_idx}: empty population. Stopping.")
            break

        bench_results, hsaco_paths = bench_perf_sweep_pipelined(
            configs=instances,
            compile_fn=compile_fn,
            repro_cmd_fn=repro_cmd_fn,
            mcpu=args.mcpu,
            num_gpus=0 if args.compile_only else args.num_gpus,
            compile_workers=args.compile_workers,
            compile_timeout=args.compile_timeout,
            post_compile_filter=post_compile_filter,
            zero_init=args.zero_init,
            iterations=args.iterations,
        )

        all_results.extend(bench_results)
        all_hsaco_paths.update(hsaco_paths)

        label_to_cfg: dict[str, dict] = {inst.label: cfg for inst, cfg in zip(instances, cfgs)}
        annotated_winners: list[dict] = []
        for cfg_inst, stats in bench_results:
            axis_dict = label_to_cfg.get(cfg_inst.label)
            if axis_dict is None or stats is None:
                continue
            annotated = dict(axis_dict)
            annotated["_tflops"] = stats.p50_tf
            annotated_winners.append(annotated)
            all_records.append(
                {
                    **{k: (list(v) if isinstance(v, tuple) else v) for k, v in axis_dict.items()},
                    "tier_idx": tier.tier_idx,
                    "tflops": stats.p50_tf,
                    "tflops_p0": stats.p0_tf,
                    "tflops_p10": stats.p10_tf,
                    "tflops_p25": stats.p25_tf,
                    "tflops_p50": stats.p50_tf,
                    "tflops_p90": stats.p90_tf,
                    "tflops_mean": stats.mean_tf,
                    "ms_stddev": stats.stddev_ms,
                    "pct_peak": stats.p50_pct,
                    "label": cfg_inst.label,
                }
            )

        if not annotated_winners:
            print(f"Tier {tier.tier_idx}: no measurable results (compile-only or all failed).")
            if args.compile_only:
                print("compile-only: skipping subsequent tiers.")
            break

        annotated_winners.sort(key=lambda d: d["_tflops"], reverse=True)
        if tier.top_k_per_stratum and stratification_key is not None:
            seen_strata: set = set()
            stratified_keepers: list[dict] = []
            for w in annotated_winners:
                key = stratification_key(w)
                if key in seen_strata:
                    continue
                seen_strata.add(key)
                stratified_keepers.append(w)
            winners = stratified_keepers[: tier.top_k_to_keep]
        else:
            winners = annotated_winners[: tier.top_k_to_keep]
        prev_winners = winners

        print(f"Tier {tier.tier_idx} top-{len(winners)}:")
        for w in winners[:5]:
            print(
                f"  {w['_tflops']:>7.1f} TF/s  "
                f"wg={w.get('wg_m')}x{w.get('wg_n')} "
                f"twg={w.get('twg_m')}x{w.get('twg_n')} "
                f"w={w.get('waves_m')}x{w.get('waves_n')} "
                f"ps={w.get('ps')} variant={w.get('variant')}"
            )

    _write_results_txt(
        out_path,
        all_records=all_records,
        bench_label=bench_label,
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
        mcpu=args.mcpu,
        seed=getattr(args, "seed", None),
        repro_cmd_fn=repro_cmd_fn,
    )
    print(f"\nTier results: {out_path} ({len(all_records)} rows)")
    if all_records:
        for line in _format_campaign_top(all_records, repro_cmd_fn, top_n=20):
            print(line)

    # Phase 3: correctness verification on the top-N across all tiers.
    if all_results and not args.compile_only:
        sorted_results = sorted(all_results, key=lambda r: (r[1].p50_tf if r[1] is not None else 0.0), reverse=True)
        verify_top_configs(
            sorted_results,
            all_hsaco_paths,
            repro_cmd_fn,
            mcpu=args.mcpu,
            top_n=min(100, len(sorted_results)),
            num_gpus=args.num_gpus,
            label=bench_label,
        )

    return out_path if all_records else None


def _build_anchor_extras(
    tier: TierSpec,
    prev_winners: list[dict],
    ambient_pins: dict,
) -> list[dict]:
    """Per-winner cross-product of `tier.axis_grid` with `tier.anchor_axes`
    pinned.

    Empty when there are no winners (tier-1) or no anchor.
    """
    if not (tier.anchor_axes and prev_winners and tier.axis_grid):
        return []

    pinned = dict(tier.anchor_axes)
    swept_axes = {k: list(v) for k, v in tier.axis_grid.items() if k not in pinned}
    swept_keys = list(swept_axes.keys())
    swept_values = [swept_axes[k] for k in swept_keys]

    extras: list[dict] = []
    for winner in prev_winners:
        base = {k: v for k, v in winner.items() if k != "_tflops"}
        for k, v in pinned.items():
            base[k] = v
        for k, v in tier.fixed_axes.items():
            base[k] = v
        for k, v in ambient_pins.items():
            base[k] = v
        if not swept_keys:
            extras.append(dict(base))
            continue
        for combo in itertools.product(*swept_values):
            cfg = dict(base)
            for k, v in zip(swept_keys, combo):
                cfg[k] = v
            extras.append(cfg)
    return extras


def _format_axes(r: dict) -> str:
    var = r.get("variant", "?")
    if isinstance(var, list) and len(var) >= 1:
        var = var[0]
    return (
        f"wg={r.get('wg_m')}x{r.get('wg_n')} "
        f"twg={r.get('twg_m')}x{r.get('twg_n')}x{r.get('twg_k')} "
        f"w={r.get('waves_m')}x{r.get('waves_n')} "
        f"occ={r.get('occ')} ps={r.get('ps')} um={r.get('unroll_factor_multiplier')} "
        f"hw={int(bool(r.get('hoist_wait')))} ll={int(bool(r.get('ll_sched')))} "
        f"rotc={int(bool(r.get('rotate_compute_stage')))} "
        f"lds={int(bool(r.get('lds_at_write')))} "
        f"epeel={int(bool(r.get('epilogue_peeling')))} "
        f"variant={var}"
    )


def _format_record_block(r: dict, rank: int, repro_cmd_fn: Callable) -> list[str]:
    tier_tag = f"T{r['tier_idx']}"
    perf = (
        f"#{rank:>2} [{tier_tag}] {r['tflops']:>7.1f} TF/s p50 "
        f"({r.get('pct_peak', 0.0):>5.1f}% peak) | "
        f"p0/p10/p25/p50/p90="
        f"{r.get('tflops_p0', 0.0):>6.1f}/"
        f"{r.get('tflops_p10', 0.0):>6.1f}/"
        f"{r.get('tflops_p25', 0.0):>6.1f}/"
        f"{r.get('tflops_p50', 0.0):>6.1f}/"
        f"{r.get('tflops_p90', 0.0):>6.1f} "
        f"mean={r.get('tflops_mean', 0.0):>6.1f} "
        f"stddev_ms={r.get('ms_stddev', 0.0):>5.3f}"
    )

    class _Stub:
        pass

    stub = _Stub()
    stub.label = r["label"]
    try:
        repro = repro_cmd_fn(stub)
    except Exception:
        repro = f"<repro unavailable> {r['label']}"
    return [perf, f"    {_format_axes(r)}", f"    repro: {repro}"]


def _format_campaign_top(all_records: list[dict], repro_cmd_fn: Callable, *, top_n: int) -> list[str]:
    top = sorted(all_records, key=lambda r: -r["tflops"])[:top_n]
    out = ["", f"Campaign top {len(top)} (sorted by p50 TF, out of {len(all_records)} measured):"]
    for i, r in enumerate(top, 1):
        out.extend(_format_record_block(r, i, repro_cmd_fn))
    return out


def _write_results_txt(
    out_path: str,
    *,
    all_records: list[dict],
    bench_label: str,
    target_m: int,
    target_n: int,
    target_k: int,
    mcpu: str,
    seed: int | None,
    repro_cmd_fn: Callable,
) -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"Bench: {bench_label}  size: m={target_m} n={target_n} k={target_k}  mcpu: {mcpu}")
    if seed is not None:
        lines.append(f"Seed: {seed}")
    lines.append(f"Total measured: {len(all_records)}")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)

    if not all_records:
        lines.append("(No measurable results.)")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    lines.extend(_format_campaign_top(all_records, repro_cmd_fn, top_n=20))

    by_tier: dict[int, list[dict]] = {}
    for r in all_records:
        by_tier.setdefault(r["tier_idx"], []).append(r)
    for tier_idx in sorted(by_tier):
        recs = sorted(by_tier[tier_idx], key=lambda r: -r["tflops"])
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"Tier {tier_idx}: {len(recs)} measured, ranked by p50 TF")
        lines.append("-" * 78)
        for i, r in enumerate(recs, 1):
            lines.extend(_format_record_block(r, i, repro_cmd_fn))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
