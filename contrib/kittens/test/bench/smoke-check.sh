#!/usr/bin/env bash
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Bench-layer smoke check: exercises sweep + single-config in --compile-only
# mode. No GPU required. Run from a worktree with the .aster venv activated.
#
# Usage:
#   bash contrib/kittens/test/bench/smoke-check.sh --mcpu gfx942
#   bash contrib/kittens/test/bench/smoke-check.sh --mcpu gfx950

set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Error: VIRTUAL_ENV must be set (activate the worktree's .aster venv first)." >&2
    exit 1
fi

MCPU=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mcpu) MCPU="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ -z "$MCPU" ]; then
    echo "Error: --mcpu required (e.g. --mcpu gfx942 or --mcpu gfx950)" >&2
    exit 1
fi

PYTHON="$(command -v python)"
HSACO_DIR="$(mktemp -d)"
trap 'rm -rf "$HSACO_DIR"' EXIT

BENCH=contrib/kittens/test/bench/bench_perf_001_gemm_fp16_weak_scaled.py

echo "=== Step 1: sweep --compile-only --compile-sample 4 (parallel compile path) ==="
$PYTHON $BENCH \
    --mcpu "$MCPU" \
    --compile-only \
    --compile-sample 4 \
    --size 4096x4096x4096 \
    --hsaco-dir "$HSACO_DIR"

# Pick one label from the .hsaco files produced by step 1.
LABEL="$(ls "$HSACO_DIR" 2>/dev/null | grep '\.hsaco$' | head -1 | sed 's/\.hsaco$//')"
if [ -z "$LABEL" ]; then
    echo "ERROR: no .hsaco produced in step 1 (HSACO_DIR=$HSACO_DIR)" >&2
    exit 1
fi
echo ""
echo "  step 1 produced $(ls "$HSACO_DIR" | grep -c '\.hsaco$') hsaco(s) in $HSACO_DIR"
echo ""

echo "=== Step 2: single-config --compile-only on label '$LABEL' ==="
$PYTHON $BENCH "$LABEL" --mcpu "$MCPU" --compile-only --hsaco "$HSACO_DIR/single.hsaco"

echo ""
echo "SMOKE CHECK OK"
