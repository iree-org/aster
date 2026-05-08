#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare BENCH_RESULT_JSON measurements against ``best_known.json`` baselines.

Two modes:

* ``--measurements <file>``: parse the JSONL, compare against baselines.
* (no ``--measurements``): regression mode. Delegate to ``perf_evaluate`` to
  recompile + re-measure every stored config, then compare. Useful for
  validating a new compiler pass: every stored top-N config is recompiled and
  measured against its baseline.

Per-label regression check: one row per stored top-N entry. A measurement
matches a row only when ``(mcpu, bench, M, N, K, label)`` agrees, so the
saved top-5 acts as a multi-point regression baseline.

Status:
    REGRESS   measured < expected * (1 - regress_margin)
    IMPROVE   measured > expected * (1 + improve_margin)
    OK        otherwise
    MISS      stored entry has no matching measurement
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

import common  # noqa: E402
import perf_argparse as cli  # noqa: E402
import perf_best_known as bk  # noqa: E402
import perf_evaluate  # noqa: E402
from perf_bench_results_io import EntryKey, max_merge_by_key_label, parse_jsonl_results  # noqa: E402


DEFAULT_IMPROVE_MARGIN = 0.03
DEFAULT_REGRESS_MARGIN = 0.05
_STATUS_RANK = {"REGRESS": 0, "MISS": 1, "IMPROVE": 2, "OK": 3}


@dataclass
class Row:
    key: EntryKey
    rank: int  # 1..TOP_N, the slot index this row represents
    label: str
    expected_tflops: float
    measured: float | None


def _status(measured: float | None, expected: float, regress_margin: float, improve_margin: float) -> str:
    if measured is None:
        return "MISS"
    if measured < expected * (1.0 - regress_margin):
        return "REGRESS"
    if measured > expected * (1.0 + improve_margin):
        return "IMPROVE"
    return "OK"


def _delta_pct(measured: float | None, expected: float) -> str:
    if measured is None or expected == 0.0:
        return "---"
    return f"{(measured - expected) / expected * 100.0:+.2f}%"


def _label_tail(label: str) -> str:
    """Strip the ``m{M}xn{N}xk{K}_`` size prefix; that info is already in the M/N/K columns."""
    parts = label.split("_", 1)
    return parts[1] if len(parts) > 1 else label


def _render(rows: list[tuple[Row, str]], dry_run: bool) -> str:
    headers = ["mcpu", "bench", "M", "N", "K", "rank", "expected", "measured", "delta_pct", "status", "label"]
    sorted_rows = sorted(
        rows,
        key=lambda rs: (
            _STATUS_RANK.get(rs[1], 5),
            rs[0].key.mcpu,
            rs[0].key.bench,
            rs[0].key.M,
            rs[0].key.N,
            rs[0].key.K,
            rs[0].rank,
        ),
    )
    body: list[list[str]] = []
    for r, st in sorted_rows:
        body.append(
            [
                r.key.mcpu,
                r.key.bench,
                str(r.key.M),
                str(r.key.N),
                str(r.key.K),
                f"#{r.rank}",
                f"{r.expected_tflops:.1f}",
                "---" if r.measured is None else f"{r.measured:.1f}",
                _delta_pct(r.measured, r.expected_tflops),
                "?" if dry_run else st,
                _label_tail(r.label),
            ]
        )
    return common.render_table(headers, body)


def _summarize(rows: list[tuple[Row, str]], dry_run: bool) -> tuple[str, int]:
    if dry_run:
        return f"{len(rows)} stored entries listed (dry-run; no measurements).", 0
    counts = {k: 0 for k in _STATUS_RANK}
    for _, st in rows:
        counts[st] += 1
    n_keys = len({(r.key.mcpu, r.key.bench, r.key.M, r.key.N, r.key.K) for r, _ in rows})
    parts = [
        f"{len(rows)} stored entries across {n_keys} (mcpu,bench,M,N,K) slot(s) checked.",
        f"{counts['REGRESS']} REGRESS",
        f"{counts['IMPROVE']} IMPROVE",
        f"{counts['OK']} OK",
    ]
    if counts["MISS"]:
        parts.append(f"{counts['MISS']} MISS")
    code = 1 if counts["REGRESS"] > 0 else (2 if counts["MISS"] > 0 else 0)
    return parts[0] + "  " + ", ".join(parts[1:]) + ".", code


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_filter_args(p)
    cli.add_best_known_arg(p)
    cli.add_compile_args(p, iterations_default=cli.DEFAULT_PERF_ITERATIONS)
    cli.add_dry_run_arg(p, help_text="Show baselines without parsing or running measurements.")
    p.add_argument(
        "--measurements",
        default=None,
        help="Path to BENCH_RESULT_JSON lines (or '-' for stdin). If omitted, "
        "perf_evaluate is invoked internally to recompile + re-measure every stored config.",
    )
    p.add_argument("--allow-missing", action="store_true", help="Treat MISS as OK (exit 0 instead of 2).")
    p.add_argument(
        "--regress-margin", type=float, default=DEFAULT_REGRESS_MARGIN, help="REGRESS if measured < expected*(1-m)."
    )
    p.add_argument(
        "--improve-margin", type=float, default=DEFAULT_IMPROVE_MARGIN, help="IMPROVE if measured > expected*(1+m)."
    )
    args = p.parse_args(argv)
    data = bk.load(args.best_known_file)

    # Pick the measurements source.
    measurements_path = args.measurements
    if not args.dry_run and measurements_path is None:
        measurements_path = os.path.join(tempfile.mkdtemp(prefix="perf_dashboard_"), "run.json")
        perf_evaluate.measure_all(
            data,
            measurements_path,
            mcpu_filter=args.mcpu,
            bench_filter=args.bench,
            iterations=args.iterations,
            compile_workers=args.compile_workers,
            hsaco_dir=args.hsaco_dir,
            num_gpus=args.num_gpus,
            compile_only=args.compile_only,
        )

    # Per-(key, label) measurements: each stored top-N entry is matched by its label.
    measurements: dict[tuple[EntryKey, str], float] = {}
    if not args.dry_run:
        stream = sys.stdin if measurements_path == "-" else open(measurements_path)
        try:
            measurements = {
                k: rec.tflops_median for k, rec in max_merge_by_key_label(parse_jsonl_results(stream)).items()
            }
        finally:
            if stream is not sys.stdin:
                stream.close()

    rows: list[tuple[Row, str]] = []
    for mcpu, bench, size_key, slot in common.iter_registry(data, mcpu_filter=args.mcpu, bench_filter=args.bench):
        if not slot:
            continue
        m, n, k = bk.parse_size_key(size_key)
        key = EntryKey(mcpu, bench, m, n, k)
        for rank, entry in enumerate(slot, start=1):
            r = Row(
                key=key,
                rank=rank,
                label=entry["label"],
                expected_tflops=entry["expected_tflops"],
                measured=measurements.get((key, entry["label"])),
            )
            rows.append((r, _status(r.measured, r.expected_tflops, args.regress_margin, args.improve_margin)))

    print(_render(rows, dry_run=args.dry_run))
    print()
    summary, code = _summarize(rows, dry_run=args.dry_run)
    print(summary)

    # If any rows IMPROVED, hint how to write the new winners back to
    # best_known.json. Only print when we actually have a measurements file
    # to feed into perf_best_known_update.
    if not args.dry_run and measurements_path and measurements_path != "-":
        n_improve = sum(1 for _, st in rows if st == "IMPROVE")
        if n_improve > 0:
            bku_py = os.path.join(os.path.dirname(__file__), "perf_best_known_update.py")
            mcpu_arg = f" --mcpu {args.mcpu}" if args.mcpu else ""
            bench_arg = f" --bench {args.bench}" if args.bench else ""
            print(
                f"\nTo write {n_improve} IMPROVE winner(s) to {args.best_known_file}:",
                file=sys.stderr,
            )
            print(
                f"  python {bku_py} --input {measurements_path}"
                f" --best-known-file {args.best_known_file}{mcpu_arg}{bench_arg} --apply",
                file=sys.stderr,
            )

    if args.allow_missing and code == 2:
        code = 0
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
