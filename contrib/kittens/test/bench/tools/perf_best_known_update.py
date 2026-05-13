#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Parse a BENCH_RESULT_JSON stream and (optionally) update ``best_known.json``.

Each ``(mcpu, bench, M, N, K)`` slot stores the top-``TOP_N`` configs sorted
by ``expected_tflops`` desc. ALL labels measured for a slot in this run are
merged in (dedup by label, sort, trim) so a single sweep can populate the
full top-N -- not just the single best label.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

sys.path.insert(0, os.path.dirname(__file__))

import common  # noqa: E402
import perf_argparse as cli  # noqa: E402
import perf_best_known as bk  # noqa: E402
from perf_bench_results_io import BenchResultRecord, EntryKey, max_merge_by_key_label, parse_jsonl_results  # noqa: E402


@dataclass(frozen=True)
class _Decision:
    """One row of the update plan, per (mcpu, bench, M, N, K) slot."""

    key: EntryKey
    action: str  # NEW | UPDATE | NO-UPDATE
    n_before: int
    n_after: int
    n_input_labels: int  # distinct labels seen in input for this slot
    top1_before: float | None
    top1_after: float
    new_slot: list[dict]  # the merged top-N (length 1..TOP_N)


def merge_records_into_slot(slot: list[dict], records: list[BenchResultRecord], *, top_n: int = bk.TOP_N) -> list[dict]:
    """Merge multiple records (one per label) into ``slot``: dedup by label, sort desc, trim to ``top_n``."""
    # Index existing slot by label so a new measurement REPLACES the stored value when better.
    by_label: dict[str, float] = {e["label"]: e["expected_tflops"] for e in slot}
    for rec in records:
        rounded = round(rec.tflops_median, 1)
        prev = by_label.get(rec.label)
        if prev is None or rounded > prev:
            by_label[rec.label] = rounded
    merged = [{"label": lab, "expected_tflops": tf} for lab, tf in by_label.items()]
    merged.sort(key=lambda e: -e["expected_tflops"])
    return merged[:top_n]


def plan_updates(
    records: Iterable[BenchResultRecord],
    data: dict,
    *,
    mcpu_filter: str | None = None,
    bench_filter: str | None = None,
) -> list[_Decision]:
    """Per slot, merge ALL measured labels into the top-N.

    Returns one decision per touched slot.
    """
    by_key_label = max_merge_by_key_label(records)
    per_key: dict[EntryKey, list[BenchResultRecord]] = defaultdict(list)
    for (key, _label), rec in by_key_label.items():
        if mcpu_filter is not None and key.mcpu != mcpu_filter:
            continue
        if bench_filter is not None and key.bench != bench_filter:
            continue
        per_key[key].append(rec)
    decisions: list[_Decision] = []
    for key in sorted(per_key.keys(), key=lambda k: (k.mcpu, k.bench, k.M, k.N, k.K)):
        slot = data.get(key.mcpu, {}).get(key.bench, {}).get(bk.size_key(key.M, key.N, key.K)) or []
        new_slot = merge_records_into_slot(slot, per_key[key])
        top1_before = slot[0]["expected_tflops"] if slot else None
        if not slot:
            action = "NEW"
        elif new_slot == slot:
            action = "NO-UPDATE"
        else:
            action = "UPDATE"
        decisions.append(
            _Decision(
                key=key,
                action=action,
                n_before=len(slot),
                n_after=len(new_slot),
                n_input_labels=len(per_key[key]),
                top1_before=top1_before,
                top1_after=new_slot[0]["expected_tflops"],
                new_slot=new_slot,
            )
        )
    return decisions


def render_table(decisions: list[_Decision]) -> str:
    headers = [
        "mcpu",
        "bench",
        "M",
        "N",
        "K",
        "top1_before",
        "top1_after",
        "delta_pct",
        "n_input",
        "n_before",
        "n_after",
        "action",
    ]
    rows: list[list[str]] = []
    for d in decisions:
        if d.top1_before is None or d.top1_before == 0:
            top1b_str = "--"
            delta_str = "--"
        else:
            top1b_str = f"{d.top1_before:.1f}"
            delta_str = f"{(d.top1_after - d.top1_before) / d.top1_before * 100.0:+.1f}%"
        rows.append(
            [
                d.key.mcpu,
                d.key.bench,
                str(d.key.M),
                str(d.key.N),
                str(d.key.K),
                top1b_str,
                f"{d.top1_after:.1f}",
                delta_str,
                str(d.n_input_labels),
                str(d.n_before),
                str(d.n_after),
                d.action,
            ]
        )
    return common.render_table(headers, rows) + "\n"


def apply_updates(
    input_stream,
    best_known_path: str = str(bk.DEFAULT_PATH),
    *,
    mcpu_filter: str | None = None,
    bench_filter: str | None = None,
    apply: bool = False,
    out_stream=None,
) -> tuple[list[_Decision], bool]:
    """Parse, plan, print, and optionally write JSON; return (decisions, wrote)."""
    if out_stream is None:
        out_stream = sys.stdout
    data = bk.load(best_known_path)
    records = list(parse_jsonl_results(input_stream))
    decisions = plan_updates(
        records,
        data,
        mcpu_filter=mcpu_filter,
        bench_filter=bench_filter,
    )
    print(render_table(decisions), file=out_stream)
    actionable = [d for d in decisions if d.action in ("NEW", "UPDATE")]
    n_new = sum(1 for d in actionable if d.action == "NEW")
    n_update = len(actionable) - n_new
    n_no = sum(1 for d in decisions if d.action == "NO-UPDATE")
    print(
        f"Input: {len(records)} BENCH_RESULT_JSON record(s) across {len(decisions)} slot(s).",
        file=out_stream,
    )
    print(
        f"Plan: {len(actionable)} writes ({n_new} NEW, {n_update} UPDATE), {n_no} NO-UPDATE.",
        file=out_stream,
    )
    short_slots = [d for d in decisions if d.n_input_labels < bk.TOP_N]
    if short_slots:
        print(
            f"NOTE: {len(short_slots)} slot(s) had < TOP_N={bk.TOP_N} distinct labels in the input "
            f"-> top-N cannot fill from this run alone. "
            f"Bottleneck is upstream (sweep / verify), not the merge.",
            file=out_stream,
        )
    if not apply or not actionable:
        return decisions, False
    for d in actionable:
        data.setdefault(d.key.mcpu, {}).setdefault(d.key.bench, {})[bk.size_key(d.key.M, d.key.N, d.key.K)] = d.new_slot
    bk.save(data, best_known_path)
    print(f"Wrote {best_known_path}", file=out_stream)
    return decisions, True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_jsonl_input_arg(p)
    cli.add_filter_args(p)
    cli.add_apply_arg(p)
    cli.add_best_known_arg(p)
    args = p.parse_args(argv)

    stream = sys.stdin if args.input == "-" else open(args.input)
    try:
        apply_updates(
            stream,
            args.best_known_file,
            mcpu_filter=args.mcpu,
            bench_filter=args.bench,
            apply=args.apply,
        )
    finally:
        if args.input != "-":
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
