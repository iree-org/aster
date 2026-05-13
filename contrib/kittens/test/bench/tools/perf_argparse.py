# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared argparse builders for ``perf_*`` tools in ``test/bench/tools``.

Every shared flag (``--mcpu``, ``--bench``, ``--best-known-file``,
``--iterations``, ``--compile-workers``, ``--hsaco-dir``, ``--num-gpus``,
``--compile-only``, ``--input``, ``--out``, ``--apply``, ``--dry-run``) has
a single definition here. Tools compose the subset they need; a flag that
is not added for a tool is simply not part of that tool's CLI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import perf_best_known as bk  # noqa: E402


# Default iteration counts shared across tools.
DEFAULT_DASHBOARD_ITERATIONS = 1000  # perf_dashboard: regression measurement tight CI
DEFAULT_COMPILE_WORKERS = 16


# ---- Filter / selector args ---------------------------------------------


def add_filter_args(
    p: argparse.ArgumentParser,
    *,
    mcpu_required: bool = False,
    bench_required: bool = False,
) -> None:
    """Add ``--mcpu`` and ``--bench``.

    ``*_required=True`` for tools that drive ONE bench / mcpu (perf_explore);
    elsewhere both are optional filters over the registry.
    """
    p.add_argument(
        "--mcpu",
        required=mcpu_required,
        default=None,
        help=(
            "Target / filter mcpu (e.g. gfx942, gfx950). "
            + ("Required." if mcpu_required else "Optional filter over the registry.")
        ),
    )
    p.add_argument(
        "--bench",
        required=bench_required,
        default=None,
        help=(
            "Bench file basename (e.g. bench_perf_102_gemm_python_multitile_lds_cdna3). "
            + ("Required." if bench_required else "Optional filter over the registry.")
        ),
    )


def add_best_known_arg(p: argparse.ArgumentParser) -> None:
    """Add ``--best-known-file`` with the default registry path."""
    p.add_argument("--best-known-file", default=str(bk.DEFAULT_PATH), help="Path to best_known.json.")


# ---- Compile / execute parallelism + artifacts --------------------------


def add_compile_args(p: argparse.ArgumentParser, *, iterations_default: int | None = None) -> None:
    """Add ``--compile-workers``, ``--hsaco-dir``, ``--num-gpus``, ``--iterations``, ``--compile-only``."""
    p.add_argument(
        "--compile-workers",
        type=int,
        default=DEFAULT_COMPILE_WORKERS,
        help=f"Parallel compile workers (default: {DEFAULT_COMPILE_WORKERS}).",
    )
    p.add_argument(
        "--hsaco-dir",
        default=None,
        help="Persistent dir for compiled .hsaco/.s artifacts. Default: ephemeral mkdtemp.",
    )
    p.add_argument(
        "--num-gpus", type=int, default=None, help="GPUs to use during execute phase (auto-detect if unset)."
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=iterations_default,
        help=(
            f"Iterations per measured config (default: {iterations_default})."
            if iterations_default is not None
            else "Iterations per measured config (bench harness default if unset)."
        ),
    )
    p.add_argument("--compile-only", action="store_true", help="Compile but skip execute + verify.")


# ---- JSONL I/O ----------------------------------------------------------


def add_jsonl_input_arg(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add ``--input <path | ->`` for JSONL of ``BENCH_RESULT_JSON`` records."""
    p.add_argument(
        "--input",
        required=required,
        default=None,
        help="JSONL file with BENCH_RESULT_JSON records (or '-' for stdin).",
    )


def add_jsonl_output_arg(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add ``--out <path>`` for JSONL of ``BENCH_RESULT_JSON`` records."""
    p.add_argument("--out", required=required, default=None, help="Output JSONL path for BENCH_RESULT_JSON lines.")


# ---- Best-known update mode --------------------------------------------


def add_apply_arg(p: argparse.ArgumentParser) -> None:
    """Add ``--apply``."""
    p.add_argument("--apply", action="store_true", help="Write changes back to best_known.json.")


# ---- Dry-run -----------------------------------------------------------


def add_dry_run_arg(p: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    """Add ``--dry-run`` with tool-specific help text."""
    p.add_argument("--dry-run", action="store_true", help=help_text or "Plan only; do not compile or write.")
