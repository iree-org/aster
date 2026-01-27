#!/bin/bash

set -e

echo $(env)

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TEST_SCRIPT="mlir_kernels/test/test_mfma_dword4_mxnxk_16x16x16_f16f16f32.py"

profile_kernel() {
    local num_workgroups="$1"
    local num_waves="$2"
    local m="$3"
    local n="$4"
    local k="$5"
    local mcpu="${6:-gfx942}"

    local machine_name="$(hostname)"
    local trace_dir="trace_${machine_name}"
    local trace="${trace_dir}_mfma_m${m}_n${n}_k${k}_wg${num_workgroups}_waves${num_waves}"

    echo ""
    echo "========================================"
    echo "Profiling: mfma_dword4_mxnxk_16x16x16_f16f16f32"
    echo "m=$m, n=$n, k=$k"
    echo "Workgroups: $num_workgroups, Waves: $num_waves"
    echo "mcpu: $mcpu"
    echo "========================================"
    echo ""

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-activity 10 \
        -d \"$trace\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" \
        --num-workgroups \"$num_workgroups\" \
        --num-waves \"$num_waves\" \
        --m \"$m\" \
        --n \"$n\" \
        --k \"$k\" \
        --mcpu \"$mcpu\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Parse command line arguments for profiling
# Default: num_workgroups=304, num_waves=1, m=3, n=3, k=16
profile_kernel "${1:-304}" "${2:-1}" "${3:-3}" "${4:-3}" "${5:-16}" "${6:-gfx942}"
