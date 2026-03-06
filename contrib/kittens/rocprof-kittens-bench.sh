#!/bin/bash
# Profile a kittens bench config with rocprofv3 ATT tracing.
#
# Compiles the HSACO first (no rocprofv3), then executes it under rocprofv3.
# This two-phase approach is required because rocprofv3 breaks when the
# process does MLIR compilation.
#
# The target script must support:
#   --compile-only --hsaco <path>   (phase 1: compile, write HSACO, exit)
#   --hsaco <path> --iterations N   (phase 2: execute pre-compiled HSACO)
#
# Usage:
#   contrib/kittens/rocprof-kittens-bench.sh <script.py> [script args...]
#
# Examples:
#   contrib/kittens/rocprof-kittens-bench.sh bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
#       --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
#       --m-tiles 2 --n-tiles 3 --k-tiles 1 --stages 2 --k-scaling-factor 256
#
#   ITERATIONS=5 PERF_COUNTERS="SQ_INSTS_VALU SQ_INSTS_VMEM" \
#       contrib/kittens/rocprof-kittens-bench.sh bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
#           --m-wg 19 --n-wg 16 --m-waves 2 --n-waves 2 \
#           --m-tiles 4 --n-tiles 4 --k-tiles 1 --stages 2 --k-scaling-factor 256
#
# Environment variables:
#   ROCPROFV3            - path to rocprofv3 (default: auto-detected via which)
#   ITERATIONS           - kernel launches for profiling (default: 5)
#   PERF_COUNTERS        - space-separated PMC counters for ATT
#   EXTRA_ROCPROFV3_ARGS - extra rocprofv3 flags (e.g. "--kernel-iteration-range [3-5]")

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test"

. "${SCRIPT_DIR}/../../mlir_kernels/nanobenchmarks/utils.sh"
check_venv

PYTHON_BIN="$(get_python_bin)"
ROCPROFV3="${ROCPROFV3:-$(which rocprofv3 2>/dev/null || true)}"
if [ -z "$ROCPROFV3" ]; then
    echo "Error: rocprofv3 not found. Set ROCPROFV3=/path/to/rocprofv3"
    exit 1
fi
ITERATIONS="${ITERATIONS:-5}"
PERF_COUNTERS="${PERF_COUNTERS:-SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_WAIT_INST_LDS}"

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $(basename "$0") <script.py> [script args...]"
    echo "  Scripts are relative to contrib/kittens/test/"
    exit 1
fi

PY_SCRIPT="${TEST_DIR}/$1"
shift

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found"
    exit 1
fi

HSACO_DIR=$(mktemp -d -t kittens_hsaco_XXXXXX)
HSACO_PATH="${HSACO_DIR}/kernel.hsaco"
TRACE_LABEL="$(basename "$PY_SCRIPT" .py)"
TRACE_DIR="$(make_trace_dir "kittens_${TRACE_LABEL}" "")"

# -- Phase 1: Compile HSACO (no rocprofv3) ------------------------------------
echo "=== Phase 1: Compile HSACO ==="
"$PYTHON_BIN" "$PY_SCRIPT" "$@" --compile-only --hsaco "$HSACO_PATH"
echo "  HSACO: $HSACO_PATH ($(ls -lh "$HSACO_PATH" | awk '{print $5}'))"

# -- Phase 2: Execute under rocprofv3 ATT -------------------------------------
echo ""
echo "=== Phase 2: rocprofv3 ATT ==="
echo "  counters: $PERF_COUNTERS"
echo "  trace:    $TRACE_DIR"

# shellcheck disable=SC2086
"$ROCPROFV3" \
    --att \
    --att-perfcounter-ctrl 10 \
    --att-perfcounters "$PERF_COUNTERS" \
    --kernel-iteration-range "[3-3]" \
    -d "$TRACE_DIR" \
    $EXTRA_ROCPROFV3_ARGS \
    -- \
    "$PYTHON_BIN" "$PY_SCRIPT" "$@" \
    --iterations "$ITERATIONS" --hsaco "$HSACO_PATH"

echo ""
echo "Done. Trace: $(pwd)/$TRACE_DIR"
echo "      HSACO: $HSACO_PATH"
