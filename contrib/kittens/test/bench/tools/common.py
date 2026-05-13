# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared helpers for perf_* tools in ``contrib/kittens/test/bench/tools``.

Path resolution, argparse boilerplate, filtered registry iteration,
fixed-width table rendering, and SIGINT-safe subprocess streaming.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterator


# ---- Path helpers ---------------------------------------------------------


def bench_dir() -> Path:
    """Directory containing bench_perf_*.py files (parent of tools/)."""
    return Path(__file__).parent.parent


def bench_file_path(bench: str) -> Path:
    """Absolute path to ``<bench>.py`` for the given bench basename."""
    return bench_dir() / f"{bench}.py"


# ---- Registry iteration ---------------------------------------------------


def iter_registry(
    data: dict,
    *,
    mcpu_filter: str | None = None,
    bench_filter: str | None = None,
) -> Iterator[tuple[str, str, str, list[dict]]]:
    """Yield ``(mcpu, bench, size_key, slot)`` tuples, applying filters."""
    for mcpu, by_bench in data.items():
        if mcpu_filter is not None and mcpu != mcpu_filter:
            continue
        for bench, by_size in by_bench.items():
            if bench_filter is not None and bench != bench_filter:
                continue
            for size_key, slot in by_size.items():
                yield mcpu, bench, size_key, slot


# ---- Table rendering ------------------------------------------------------


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render fixed-width columns with a separator line under the header."""
    grid = [headers] + rows
    widths = [max(len(r[c]) for r in grid) for c in range(len(headers))]
    out = ["  ".join(r[c].ljust(widths[c]) for c in range(len(headers))) for r in grid]
    out.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(out)


# ---- Subprocess streaming -------------------------------------------------


def stream_subprocess(cmd: list[str], *, label: str = "child", grace_sec: float = 60.0) -> int:
    """Run ``cmd``, stream its stdout/stderr in real time, forward Ctrl+C as SIGINT.

    Returns the child's exit code. On ``KeyboardInterrupt`` the child gets
    SIGINT and up to ``grace_sec`` to drain; a second Ctrl+C escalates to
    SIGTERM, then SIGKILL on timeout. The KeyboardInterrupt is re-raised
    after the child has exited.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    out = sys.stdout.buffer

    def _drain() -> None:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                return
            out.write(chunk)
            out.flush()

    try:
        _drain()
    except KeyboardInterrupt:
        print(f"\n[Ctrl+C: forwarding SIGINT to {label}; waiting for graceful drain]", file=sys.stderr)
        proc.send_signal(signal.SIGINT)
        try:
            _drain()
        except KeyboardInterrupt:
            print("\n[Second Ctrl+C; SIGTERM]", file=sys.stderr)
            proc.terminate()
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            print(f"[{label} did not exit within {grace_sec}s; SIGKILL]", file=sys.stderr)
            proc.kill()
            proc.wait()
        raise
    return proc.wait()
