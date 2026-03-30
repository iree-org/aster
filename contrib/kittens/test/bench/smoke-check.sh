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
HSACO_DIR=$(mktemp -d)

BENCH=contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py
ARGS="--m-wg 19 --n-wg 16 --m-waves 2 --n-waves 2 \
    --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --stages 4 --k-scaling-factor 256"
HSACO="$HSACO_DIR/smoke.hsaco"

echo "=== Step 1: compile only, produce HSACO ==="
$PYTHON $BENCH $ARGS --compile-only --hsaco "$HSACO"
echo "PASS"

echo ""
echo "=== Step 2: execute pre-compiled HSACO ==="
$PYTHON $BENCH $ARGS --hsaco "$HSACO"
echo "PASS"

echo ""
echo "=== Step 3: compile + run (no pre-compiled HSACO) ==="
$PYTHON $BENCH $ARGS
echo "PASS"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "All smoke checks passed."
rm -rf "$HSACO_DIR"

echo ""
echo "To run the full benchmark sweep (expensive, requires GPU):"
echo "  $PYTHON $BENCH"
echo ""
echo "Options: --num-gpus N (default: auto-detect), --compile-workers N (default: 8)"
