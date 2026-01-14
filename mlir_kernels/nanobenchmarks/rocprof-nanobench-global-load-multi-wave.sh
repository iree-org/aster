#!/bin/bash

set -e

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_global_load_multi_wave.py"

profile_kernel() {
    local num_iters="$1"
    local num_kernel_runs="$2"
    local num_tiles="$3"
    local num_waves="$4"
    local num_cus="$5"

    local machine_name="$(hostname)"
    local trace="trace_${machine_name}_nanobench_global_load_multi_wave_iters${num_iters}_runs${num_kernel_runs}_cus${num_cus}"

    echo ""
    echo "========================================"
    echo "Profiling: nanobench_global_load_multi_wave"
    echo "num_iters=$num_iters, num_kernel_runs=$num_kernel_runs, num_cus=$num_cus"
    echo "4 waves per block (256 threads)"
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
        --num-tiles \"$num_tiles\" \
        --num-waves \"$num_waves\" \
        --num-cus \"$num_cus\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Default: num_iters=10, num_kernel_runs=10, num_tiles=16, num_cus=304
# Note: 16 tiles per CU over 4 waves = 4 tiles per SIMD
profile_kernel "${1:-10}" "${2:-10}" "${3:-16}" "${4:-4}" "${5:-304}"
