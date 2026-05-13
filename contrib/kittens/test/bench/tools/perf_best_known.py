# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Load / save the best-known config registry (``best_known.json``).

The bench key is the exact basename of the bench_perf_*.py driver. Each
size slot stores the top-``TOP_N`` configs sorted by ``expected_tflops``
desc; index 0 is the canonical winner. Storage::

    {mcpu: {bench: {"MxNxK": [{label, expected_tflops}, ...]}}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

DEFAULT_PATH = Path(__file__).parent / "best_known.json"
TOP_N = 5


class Entry(TypedDict):
    label: str
    expected_tflops: float


def size_key(M: int, N: int, K: int) -> str:
    return f"{M}x{N}x{K}"


def parse_size_key(key: str) -> tuple[int, int, int]:
    m, n, k = key.split("x")
    return int(m), int(n), int(k)


def load(path: str | os.PathLike = DEFAULT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def save(data: dict, path: str | os.PathLike = DEFAULT_PATH) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def best_known(mcpu: str, bench: str, M: int, N: int, K: int) -> str | None:
    """Return the top-1 config label for (mcpu, bench, M, N, K), or None."""
    slot = load().get(mcpu, {}).get(bench, {}).get(size_key(M, N, K)) or []
    return slot[0]["label"] if slot else None


def best_known_perf(mcpu: str, bench: str, M: int, N: int, K: int) -> Entry | None:
    """Return the top-1 entry (label + expected_tflops) or None."""
    slot = load().get(mcpu, {}).get(bench, {}).get(size_key(M, N, K)) or []
    return slot[0] if slot else None


def best_known_top_n(mcpu: str, bench: str, M: int, N: int, K: int) -> list[Entry]:
    """Return the full top-N list (up to ``TOP_N`` entries, sorted desc)."""
    return list(load().get(mcpu, {}).get(bench, {}).get(size_key(M, N, K)) or [])
