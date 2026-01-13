#!/bin/bash

set -e

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_global_load_multi_tile.py"

profile_kernel() {
    local num_iters="$1"
    local num_kernel_runs="$2"
    local num_cus="$3"

    local machine_name="$(hostname)"
    local trace="trace_${machine_name}_nanobench_global_load_iters${num_iters}_runs${num_kernel_runs}_cus${num_cus}"

    echo ""
    echo "========================================"
    echo "Profiling: nanobench_global_load_multi_tile"
    echo "num_iters=$num_iters, num_kernel_runs=$num_kernel_runs, num_cus=$num_cus"
    echo "========================================"
    echo ""

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        --att-perfcounters \"TCC_HIT TCC_MISS TCC_READ TCC_WRITE\" \
        -d \"$trace\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" \
        --num-iters \"$num_iters\" \
        --num-kernel-runs \"$num_kernel_runs\" \
        --num-cus \"$num_cus\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Default: num_iters=32, num_kernel_runs=10, num_cus=304
profile_kernel "${1:-32}" "${2:-10}" "${3:-304}"
