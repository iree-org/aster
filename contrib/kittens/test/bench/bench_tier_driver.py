# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tiered successive-halving sweep driver."""

from __future__ import annotations

import contextlib
import itertools
import os
import random
import signal
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from bench_harness import bench_perf_sweep_pipelined  # noqa: E402
from perf_bench_results_io import emit_one_result  # noqa: E402
from bench_search import verify_top_configs  # noqa: E402
from bench_tier_schedule import TierSpec, apply_tier_overrides  # noqa: E302, E402


# Tier-to-tier survival rate: keep the top 10% of measured winners.
TIER_KEEP_PCT = 0.10


def _alarm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"tier time budget exceeded (signal {signum})")


@contextlib.contextmanager
def _tier_alarm(budget: float | None):
    """Arm ``signal.alarm(budget)`` for the duration; SIGALRM -> KeyboardInterrupt."""
    if not budget or budget <= 0:
        yield
        return
    old = signal.signal(signal.SIGALRM, _alarm_to_keyboard_interrupt)
    signal.alarm(int(budget))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


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
    """Walk `tier_schedule`. Each tier samples, runs, keeps top 10% to seed the next.

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
    timestamp = int(time.time())
    out_path = os.path.join(out_dir, f"tier_results_{bench_label}_{timestamp}.txt")
    top20_path = os.path.join(out_dir, f"campaign_top20_{bench_label}_{timestamp}.txt")

    # Tier-1's effective axis values seed ordinal-neighbor expansion in later tiers.
    tier1_universe: dict = {}

    for tier in schedule:
        cap = tier.max_configs
        if tier.random_seed is not None:
            random.seed(tier.random_seed)
        print(
            f"\n=== Tier {tier.tier_idx}: cap={cap}, keep_pct={TIER_KEEP_PCT * 100:.0f}%, "
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

        if getattr(args, "dry_run", False) and tier.tier_idx == 1:
            print(f"\n=== Tier 1 dry-run: {len(instances)} candidate(s) ===")
            for inst in instances:
                print(f"  {inst.label}")
            print(f"=== {len(instances)} candidate(s); exiting (--dry-run) ===")
            return None

        if not instances:
            print(f"Tier {tier.tier_idx}: empty population. Stopping.")
            break

        tier_budget = getattr(args, "tier_time_budget", None)
        with _tier_alarm(tier_budget):
            try:
                bench_results, hsaco_paths = bench_perf_sweep_pipelined(
                    configs=instances,
                    compile_fn=compile_fn,
                    repro_cmd_fn=repro_cmd_fn,
                    mcpu=args.mcpu,
                    bench=bench_label,
                    num_gpus=0 if args.compile_only else args.num_gpus,
                    compile_workers=args.compile_workers,
                    compile_timeout=args.compile_timeout,
                    post_compile_filter=post_compile_filter,
                    init_mode=args.init,
                    iterations=args.iterations,
                    hsaco_dir=getattr(args, "hsaco_dir", None),
                    results_file=getattr(args, "results_file", None),
                )
            except KeyboardInterrupt:
                # SIGALRM-induced or user-induced; bench_perf_sweep_pipelined handles
                # the in-flight drain internally and re-raises if not fully done.
                print(
                    f"\n[Tier {tier.tier_idx} interrupted (budget={tier_budget}s); moving on with partial results]",
                    file=sys.stderr,
                )
                bench_results, hsaco_paths = [], {}

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
                    "tf_stddev": stats.stddev_tf,
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
        keep_n = max(1, int(len(annotated_winners) * TIER_KEEP_PCT))
        if tier.per_stratum_diversity and stratification_key is not None:
            seen_strata: set = set()
            stratified_keepers: list[dict] = []
            for w in annotated_winners:
                key = stratification_key(w)
                if key in seen_strata:
                    continue
                seen_strata.add(key)
                stratified_keepers.append(w)
            winners = stratified_keepers[:keep_n]
        else:
            winners = annotated_winners[:keep_n]
        prev_winners = winners

        print(f"Tier {tier.tier_idx} kept {len(winners)}/{len(annotated_winners)} (top {TIER_KEEP_PCT * 100:.0f}%):")
        for w in winners[:5]:
            print(
                f"  {w['_tflops']:>7.1f} TF/s  "
                f"wg={w.get('wg_m')}x{w.get('wg_n')} "
                f"twg={w.get('twg_m')}x{w.get('twg_n')}x{w.get('twg_k')} "
                f"w={w.get('waves_m')}x{w.get('waves_n')} "
                f"ps={w.get('ps')}"
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
        _write_campaign_top_txt(
            top20_path,
            all_records=all_records,
            bench_label=bench_label,
            target_m=target_m,
            target_n=target_n,
            target_k=target_k,
            mcpu=args.mcpu,
            repro_cmd_fn=repro_cmd_fn,
            top_n=20,
        )

    # Phase 3: correctness verification on the top-N across all tiers.
    # Only verified labels are emitted as BENCH_RESULT_JSON -> best_known never
    # consumes a config that failed the numpy reference check.
    verified_labels: set[str] = set()
    if all_results and not args.compile_only:
        sorted_results = sorted(all_results, key=lambda r: (r[1].p50_tf if r[1] is not None else 0.0), reverse=True)
        verified_labels = verify_top_configs(
            sorted_results,
            all_hsaco_paths,
            repro_cmd_fn,
            mcpu=args.mcpu,
            top_n=min(100, len(sorted_results)),
            num_gpus=args.num_gpus,
            label=bench_label,
        )

    results_file = getattr(args, "results_file", None)
    if results_file and verified_labels:
        with open(results_file, "a") as rs:
            for cfg, stats in all_results:
                if stats is None or cfg.label not in verified_labels:
                    continue
                emit_one_result(rs, bench_label, cfg, stats)
        print(f"  Emitted {len(verified_labels)} verified result(s) to {results_file}")

    # Final summary block: printed AFTER verify_top_configs so the file
    # paths are the last thing on screen and easy to grab.
    if all_records:
        print()
        print("=" * 78)
        print(f"  Campaign top 20:  {top20_path}")
        print(f"  Full tier results: {out_path}")
        print("=" * 78)

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
    return (
        f"wg={r.get('wg_m')}x{r.get('wg_n')} "
        f"twg={r.get('twg_m')}x{r.get('twg_n')}x{r.get('twg_k')} "
        f"w={r.get('waves_m')}x{r.get('waves_n')} "
        f"occ={r.get('occ')} ps={r.get('ps')} um={r.get('unroll_factor_multiplier')} "
        f"hw={int(bool(r.get('hoist_wait')))} ll={int(r.get('ll_sched') or 0)} "
        f"rotc={int(bool(r.get('rotate_compute_stage')))} "
        f"epeel={int(bool(r.get('epilogue_peeling')))}"
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
        f"stddev_tf={r.get('tf_stddev', 0.0):>5.1f}"
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


def _write_campaign_top_txt(
    top_path: str,
    *,
    all_records: list[dict],
    bench_label: str,
    target_m: int,
    target_n: int,
    target_k: int,
    mcpu: str,
    repro_cmd_fn: Callable,
    top_n: int = 20,
) -> None:
    lines: list[str] = [
        "=" * 78,
        f"Bench: {bench_label}  size: m={target_m} n={target_n} k={target_k}  mcpu: {mcpu}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
    ]
    lines.extend(_format_campaign_top(all_records, repro_cmd_fn, top_n=top_n))
    with open(top_path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
