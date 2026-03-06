#!/bin/bash
# Smoke-check for bench_perf_sweep_001_gemm_fp16_weak_scaled.py
# Verifies that all 3 single-config CLI modes work end-to-end.
# Steps 1-2 are compile-only (no GPU). Step 3 requires a GPU.
#
# Usage:
#   bash contrib/kittens/test/bench/smoke-check.sh

set -euo pipefail

if [ -z "VIRTUAL_ENVT:-}" ]; then
    echo "Error: VIRTUAL_ENV must be set."
    exit 1
fi

PYTHON=$(which python)
BENCH=contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py
HSACO_TMP=$(mktemp -d)/smoke.hsaco

COMMON_ARGS="--m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
    --m-tiles 2 --n-tiles 3 --k-tiles 1 --stages 4 --k-scaling-factor 256"

echo "=== Step 1: compile only, produce HSACO ==="
$PYTHON $BENCH $COMMON_ARGS --compile-only --hsaco "$HSACO_TMP"
echo "PASS"

echo ""
echo "=== Step 2: execute pre-compiled HSACO ==="
$PYTHON $BENCH $COMMON_ARGS --hsaco "$HSACO_TMP"
echo "PASS"

echo ""
echo "=== Step 3: compile + run (no pre-compiled HSACO) ==="
$PYTHON $BENCH $COMMON_ARGS
echo "PASS"

echo ""
echo "All smoke checks passed."
rm -rf "$(dirname "$HSACO_TMP")"

echo ""
echo "To run the full benchmark sweep (expensive, requires GPU):"
echo "  $PYTHON $BENCH --sweep            # TOP_K configs only"
echo "  $PYTHON $BENCH --full-sweep       # all configs in the grid"
echo ""
echo "Options: --num-gpus N (default: auto-detect), --compile-workers N (default: 8)"
