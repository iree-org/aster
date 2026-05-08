#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Recompile + measure every stored config in best_known.json.

Imports each bench module (``contrib/kittens/test/bench/<bench>.py``) and
drives ``bench_perf_sweep_pipelined`` directly with the flat list of configs
deserialized from stored labels. The bench's existing parallel compile +
GPU-queue path is reused as-is. Each module exposes a ``BENCH_HOOKS`` dict
with ``bench_label``, ``from_label``, ``compile_fn``, ``repro_cmd_fn``,
``post_compile_filter`` for this purpose.

The output JSONL is consumed by ``perf_dashboard.py --measurements <out>``
for a per-label regression check.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Mirror the sys.path setup every bench_perf_*.py does at its top, plus our
# tools/ dir. Order matters: deeper dirs first so local modules win.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # tools/  -> common, perf_*
sys.path.insert(0, str(_HERE.parent))  # bench/  -> bench_harness, bench_search, bench_perf_*
sys.path.insert(0, str(_HERE.parent.parent))  # test/   (bench's `..`)
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))  # contrib/ -> kittens.*
sys.path.insert(0, str(_HERE.parent.parent.parent.parent.parent))  # repo root

import common  # noqa: E402
import perf_argparse as cli  # noqa: E402
import perf_best_known as bk  # noqa: E402
from perf_bench_results_io import emit_one_result  # noqa: E402


def collect_todo(
    data: dict, mcpu_filter: str | None = None, bench_filter: str | None = None
) -> dict[tuple[str, str], list[str]]:
    """Return ``{(mcpu, bench): [label, ...]}`` for every stored config (after filters)."""
    by_bench: dict[tuple[str, str], list[str]] = {}
    for mcpu, bench, _size_key, slot in common.iter_registry(data, mcpu_filter=mcpu_filter, bench_filter=bench_filter):
        for entry in slot:
            by_bench.setdefault((mcpu, bench), []).append(entry["label"])
    return by_bench


def measure_all(
    data: dict,
    out_path: str,
    *,
    mcpu_filter: str | None = None,
    bench_filter: str | None = None,
    iterations: int | None = None,
    compile_workers: int | None = None,
    hsaco_dir: str | None = None,
    num_gpus: int | None = None,
    compile_only: bool = False,
) -> str:
    """Drive ``bench_perf_sweep_pipelined`` per ``(mcpu, bench)`` group; emit verified results.

    Truncates ``out_path``, then appends one ``BENCH_RESULT_JSON`` line per
    verified config. Returns ``out_path``.
    """
    # bench_perf_sweep_pipelined + verify_top_configs live in the bench/ dir,
    # not tools/. perf_evaluate is run from tools/ so the parent path is added at import.
    from bench_harness import bench_perf_sweep_pipelined
    from bench_search import verify_top_configs

    open(out_path, "w").close()  # truncate
    by_bench = collect_todo(data, mcpu_filter, bench_filter)
    total_labels = sum(len(v) for v in by_bench.values())
    print(
        f"=== perf_evaluate: {total_labels} stored config(s) across {len(by_bench)} bench-group(s) -> {out_path} ===",
        file=sys.stderr,
    )
    for i, ((mcpu, bench), labels) in enumerate(sorted(by_bench.items()), start=1):
        print(
            f"\n  [{i}/{len(by_bench)}] {bench} (mcpu={mcpu}, {len(labels)} label(s))",
            file=sys.stderr,
        )
        try:
            mod = importlib.import_module(bench)
        except ModuleNotFoundError as e:
            print(f"    SKIP -- cannot import {bench}: {e}", file=sys.stderr)
            continue
        hooks = getattr(mod, "BENCH_HOOKS", None)
        if hooks is None:
            print(f"    SKIP -- {bench} has no BENCH_HOOKS dict", file=sys.stderr)
            continue
        configs = [hooks["from_label"](lbl, mcpu) for lbl in labels]
        per_bench_hsaco = os.path.join(hsaco_dir, bench) if hsaco_dir else None
        if per_bench_hsaco:
            os.makedirs(per_bench_hsaco, exist_ok=True)
        try:
            results, hsaco_paths = bench_perf_sweep_pipelined(
                configs=configs,
                compile_fn=hooks["compile_fn"],
                repro_cmd_fn=hooks["repro_cmd_fn"],
                mcpu=mcpu,
                bench=hooks["bench_label"],
                num_gpus=0 if compile_only else num_gpus,
                compile_workers=compile_workers,
                post_compile_filter=hooks["post_compile_filter"],
                iterations=iterations,
                hsaco_dir=per_bench_hsaco,
                results_file=None,  # we emit after verification, not during sweep
            )
        except KeyboardInterrupt:
            print("\n[perf_evaluate: Ctrl+C; partial measurements stop here]", file=sys.stderr)
            raise

        # Verify on GPU; only verified labels are emitted.
        verified: set[str] = set()
        if results and not compile_only:
            sorted_results = sorted(results, key=lambda r: (r[1].p50_tf if r[1] is not None else 0.0), reverse=True)
            verified = verify_top_configs(
                sorted_results,
                hsaco_paths,
                hooks["repro_cmd_fn"],
                mcpu=mcpu,
                top_n=len(sorted_results),
                num_gpus=num_gpus,
                label=hooks["bench_label"],
            )
        n_total = len(configs)
        if verified:
            with open(out_path, "a") as rs:
                for cfg, stats in results:
                    if stats is None or cfg.label not in verified:
                        continue
                    emit_one_result(rs, hooks["bench_label"], cfg, stats)
            print(
                f"    Emitted {len(verified)}/{n_total} verified result(s) to {out_path}",
                file=sys.stderr,
            )
        elif not compile_only:
            print(
                f"    WARNING: {len(verified)}/{n_total} configs passed correctness for {bench}",
                file=sys.stderr,
            )
    print(f"=== perf_evaluate done -> {out_path} ===", file=sys.stderr)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_filter_args(p)
    cli.add_best_known_arg(p)
    cli.add_compile_args(p, iterations_default=cli.DEFAULT_PERF_ITERATIONS)
    cli.add_jsonl_output_arg(p)
    args = p.parse_args(argv)
    data = bk.load(args.best_known_file)
    measure_all(
        data,
        args.out,
        mcpu_filter=args.mcpu,
        bench_filter=args.bench,
        iterations=args.iterations,
        compile_workers=args.compile_workers,
        hsaco_dir=args.hsaco_dir,
        num_gpus=args.num_gpus,
        compile_only=args.compile_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
