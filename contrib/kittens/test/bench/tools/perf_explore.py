#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep a bench_perf_xxx over CLI-specified (M, N, K) sizes; update best_known.json.

Sizes are required and given via one or more ``--size MxNxK`` flags
(repeat for multiple). There is no default -- pass the sizes you want.

Usage:
    python perf_explore.py --bench bench_perf_102_gemm_python_multitile_lds_cdna3 \\
        --mcpu gfx942 --size 2432x4096x4096 --size 4864x4096x4096 \\
        --compile-sample 100                            # dry-run (print plan)

    python perf_explore.py --bench ... --mcpu ... --size 2048x4096x4096 --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # tools/    -> common, perf_*
sys.path.insert(0, str(_HERE.parent))  # bench/    -> bench_search, bench_perf_*
sys.path.insert(0, str(_HERE.parent.parent))  # test/     (bench's `..`)
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))  # contrib/  -> kittens.*
sys.path.insert(0, str(_HERE.parent.parent.parent.parent.parent))  # repo root

import common  # noqa: E402
import perf_argparse as cli  # noqa: E402
import perf_best_known_update as bku  # noqa: E402
from bench_search import parse_mnk  # noqa: E402  (shared MxNxK grammar)


def _size_arg(s: str) -> tuple[int, int, int]:
    """Argparse type wrapper around the shared ``parse_mnk`` grammar."""
    try:
        return parse_mnk(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _run_bench(bench: str, mcpu: str, M: int, N: int, K: int, extra: list[str], out_path: str) -> None:
    """Stream ``bench_perf_xxx --size MxNxK`` to stdout; child writes its own JSON to ``out_path``."""
    bench_path = common.bench_file_path(bench)
    if not bench_path.is_file():
        raise FileNotFoundError(f"No such bench file: {bench_path}")
    cmd = [
        sys.executable,
        "-u",
        str(bench_path),
        "--mcpu",
        mcpu,
        "--size",
        f"{M}x{N}x{K}",
        "--results-file",
        out_path,
        *extra,
    ]
    print(f"\n=== {bench} {M}x{N}x{K} (mcpu={mcpu}) ===", file=sys.stderr)
    print("  " + " ".join(cmd), file=sys.stderr)
    rc = common.stream_subprocess(cmd, label=f"{bench} {M}x{N}x{K}")
    if rc != 0:
        raise RuntimeError(f"{bench} {M}x{N}x{K} returned exit code {rc}")


def _fmt_size(s: tuple[int, int, int]) -> str:
    return f"{s[0]}x{s[1]}x{s[2]}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_filter_args(p, mcpu_required=True, bench_required=True)
    cli.add_best_known_arg(p)
    cli.add_compile_args(p, iterations_default=cli.DEFAULT_PERF_ITERATIONS)
    cli.add_apply_arg(p)
    cli.add_dry_run_arg(
        p,
        help_text="Forward to bench: print tier-1 candidates per size and exit. No compile or run.",
    )
    p.add_argument(
        "--size",
        dest="sizes",
        action="append",
        required=True,
        type=_size_arg,
        metavar="MxNxK",
        help="Exploration size; repeat for multiple. Required, no default.",
    )
    p.add_argument("--compile-sample", type=int, default=100, help="Per-tier compile-sample cap (forwarded).")
    p.add_argument("--seed", type=int, default=1, help="Random seed for tier sampling (forwarded).")
    p.add_argument(
        "--out",
        default=None,
        help="Aggregated JSON path; accumulates across sizes. Default: <hsaco-dir>/run.json.",
    )
    p.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Per-TIER wall-clock seconds (forwarded to bench --tier-time-budget). "
        "Bench raises KeyboardInterrupt on expiry and drains gracefully. Default: unlimited.",
    )
    args = p.parse_args(argv)

    hsaco_dir = args.hsaco_dir or tempfile.mkdtemp(prefix="perf_explore_")
    if os.path.exists(hsaco_dir):
        shutil.rmtree(hsaco_dir)
    os.makedirs(hsaco_dir)
    out_path = args.out or os.path.join(hsaco_dir, "run.json")
    # Truncate so SIZES accumulate cleanly across this invocation.
    open(out_path, "w").close()

    extra = ["--compile-sample", str(args.compile_sample), "--seed", str(args.seed), "--hsaco-dir", hsaco_dir]
    if args.compile_workers is not None:
        extra += ["--compile-workers", str(args.compile_workers)]
    if args.iterations is not None:
        extra += ["--iterations", str(args.iterations)]
    if args.num_gpus is not None:
        extra += ["--num-gpus", str(args.num_gpus)]
    if args.compile_only:
        extra += ["--compile-only"]
    if args.time_budget is not None:
        extra += ["--tier-time-budget", str(args.time_budget)]
    if args.dry_run:
        extra += ["--dry-run"]

    # ---- Banner ----
    print("\n=== perf_explore ===", file=sys.stderr)
    print(f"  bench           = {args.bench}", file=sys.stderr)
    print(f"  mcpu            = {args.mcpu}", file=sys.stderr)
    print(f"  compile-sample  = {args.compile_sample}", file=sys.stderr)
    print(
        f"  compile-workers = {args.compile_workers if args.compile_workers is not None else '(child default)'}",
        file=sys.stderr,
    )
    print(f"  seed            = {args.seed}", file=sys.stderr)
    print(f"  hsaco-dir       = {hsaco_dir} (cleared)", file=sys.stderr)
    print(f"  out             = {out_path} (truncated)", file=sys.stderr)
    print(
        f"  time-budget     = {f'{args.time_budget:.0f}s per tier' if args.time_budget else 'unlimited'}",
        file=sys.stderr,
    )
    print(f"  best-known-file = {args.best_known_file}", file=sys.stderr)
    print(f"  apply           = {args.apply}", file=sys.stderr)
    sizes = args.sizes
    print(f"  sizes           = {', '.join(_fmt_size(s) for s in sizes)}", file=sys.stderr)

    # ---- Run ----
    start = time.monotonic()
    interrupted = False
    sizes_completed = 0
    try:
        for i, (M, N, K) in enumerate(sizes, start=1):
            elapsed = time.monotonic() - start
            print(f"\n[{i}/{len(sizes)}] {M}x{N}x{K}  (cumulative elapsed {elapsed:.0f}s)", file=sys.stderr)
            _run_bench(args.bench, args.mcpu, M, N, K, extra, out_path)
            sizes_completed = i
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n[Ctrl+C: stopped after {sizes_completed}/{len(sizes)} sizes; processing partial results]",
            file=sys.stderr,
        )

    # ---- Update primitive (reads the aggregated .json file) ----
    with open(out_path) as f:
        decisions, wrote = bku.apply_updates(
            f,
            args.best_known_file,
            mcpu_filter=args.mcpu,
            bench_filter=args.bench,
            apply=args.apply,
        )

    # ---- Footer + next-step hints ----
    total = time.monotonic() - start
    actionable = [d for d in decisions if d.action in ("NEW", "UPDATE")]
    here = os.path.dirname(__file__)
    bku_py = os.path.join(here, "perf_best_known_update.py")
    dash_py = os.path.join(here, "perf_dashboard.py")
    print(
        f"\n=== perf_explore {'interrupted' if interrupted else 'done'}: "
        f"{sizes_completed}/{len(sizes)} sizes in {total:.0f}s; "
        f"{len(actionable)} actionable, wrote={wrote} ===",
        file=sys.stderr,
    )
    print(f"\nMeasurements: {out_path}", file=sys.stderr)
    if actionable and not args.apply:
        print(f"\nTo write {len(actionable)} winner(s) to {args.best_known_file}:", file=sys.stderr)
        print(
            f"  python {bku_py} --input {out_path} --mcpu {args.mcpu} --bench {args.bench} --apply",
            file=sys.stderr,
        )
    elif wrote:
        print(f"\nWrote winners to {args.best_known_file}.", file=sys.stderr)
    print("\nTo compare a fresh run against baselines (regression check):", file=sys.stderr)
    print(
        f"  python {dash_py} --measurements {out_path} --mcpu {args.mcpu} --bench {args.bench}",
        file=sys.stderr,
    )
    print(f"\nTo inspect the registry: cat {args.best_known_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
