# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""BENCH_RESULT_JSON contract: parse + max-merge + emit one record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator


RESULT_SENTINEL = "BENCH_RESULT_JSON:"


@dataclass(frozen=True)
class EntryKey:
    """Identity of a (problem, target) measurement slot."""

    mcpu: str
    bench: str
    M: int
    N: int
    K: int


@dataclass(frozen=True)
class BenchResultRecord:
    """One measured config from one harness run.

    ``tflops_median`` is the headline.
    """

    key: EntryKey
    label: str
    tflops_median: float
    tflops_p10: float | None = None
    tflops_p90: float | None = None
    tflops_mean: float | None = None
    sample_n: int = 1


def parse_jsonl_results(stream: Iterable[str]) -> Iterator[BenchResultRecord]:
    """Yield one BenchResultRecord per valid BENCH_RESULT_JSON line; skip noise."""
    for raw in stream:
        idx = raw.find(RESULT_SENTINEL)
        if idx < 0:
            continue
        payload = raw[idx + len(RESULT_SENTINEL) :].strip()
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        try:
            key = EntryKey(
                mcpu=d["mcpu"],
                bench=d["bench"],
                M=int(d["M"]),
                N=int(d["N"]),
                K=int(d["K"]),
            )
            tf_median = float(d["tflops_median"])
            label = d["label"]
        except (KeyError, TypeError, ValueError):
            continue
        yield BenchResultRecord(
            key=key,
            label=label,
            tflops_median=tf_median,
            tflops_p10=_as_optional_float(d.get("tflops_p10")),
            tflops_p90=_as_optional_float(d.get("tflops_p90")),
            tflops_mean=_as_optional_float(d.get("tflops_mean")),
            sample_n=int(d.get("sample_n", 1)),
        )


def _as_optional_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def max_merge_by_key_label(
    records: Iterable[BenchResultRecord],
) -> dict[tuple[EntryKey, str], BenchResultRecord]:
    """Return the best record per ``(EntryKey, label)`` (max tflops_median across runs)."""
    by_key_label: dict[tuple[EntryKey, str], BenchResultRecord] = {}
    for r in records:
        k = (r.key, r.label)
        prev = by_key_label.get(k)
        if prev is None or r.tflops_median > prev.tflops_median:
            by_key_label[k] = r
    return by_key_label


def max_merge_by_key(
    records: Iterable[BenchResultRecord],
) -> dict[EntryKey, BenchResultRecord]:
    """Return the best record per ``EntryKey`` (max tflops_median across runs and labels)."""
    best_per_key: dict[EntryKey, BenchResultRecord] = {}
    for (key, _label), r in max_merge_by_key_label(records).items():
        prev = best_per_key.get(key)
        if prev is None or r.tflops_median > prev.tflops_median:
            best_per_key[key] = r
    return best_per_key


def emit_one_result(stream, bench: str, cfg, stats) -> None:
    """Write one BENCH_RESULT_JSON line to ``stream``.

    ``stream=None`` discards.
    """
    if stream is None:
        return
    m, n, k = cfg.spec.gemm_size
    print(
        RESULT_SENTINEL
        + json.dumps(
            {
                "mcpu": cfg.mapping.mcpu,
                "bench": bench,
                "M": m,
                "N": n,
                "K": k,
                "label": cfg.label,
                "tflops_median": stats.p50_tf,
                "tflops_p10": stats.p10_tf,
                "tflops_p90": stats.p90_tf,
                "tflops_mean": stats.mean_tf,
                "tflops_p0": stats.p0_tf,
                "tflops_p25": stats.p25_tf,
                "tflops_stddev": stats.stddev_tf,
                "pct_peak_p50": stats.p50_pct,
                "sample_n": 1,
            }
        ),
        file=stream,
    )
    stream.flush()
