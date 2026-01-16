#!/bin/bash

set -e

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_global_load.py"

profile_kernel() {
    local num_iters="$1"
    local num_kernel_runs="$2"
    local num_tiles="$3"
    local num_waves="$4"
    local num_cus="$5"

    local machine_name="$(hostname)"
    local trace="trace_${machine_name}_nanobench_global_load${num_iters}_runs${num_kernel_runs}_num_tiles${num_tiles}_num_waves${num_waves}_num_cus${num_cus}"

    echo ""
    echo "========================================"
    echo "Profiling: nanobench_global_load"
    echo "num_iters=$num_iters, num_kernel_runs=$num_kernel_runs"
    echo "num_tiles=$num_tiles, num_waves=$num_waves, num_cus=$num_cus"
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

# 16 tiles per workgroup and 1216 workgroups 1 wave = **the same** 4 tiles per SIMD
profile_kernel "${1:-10}" "${2:-10}" "${3:-16}" "${4:-1}" "${5:-1216}"

# 16 tiles per per workgroup and 304 workgroups 4 waves = **the same** 4 tiles per SIMD
profile_kernel "${1:-10}" "${2:-10}" "${3:-16}" "${4:-4}" "${5:-304}"
