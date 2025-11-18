#!/bin/bash

set -e

echo $(env)

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TEST_SCRIPT="mlir_kernels/test/test_copy_1d.py"

profile_kernel() {
    local workgroups="$1"
    local waves="$2"
    local elements_per_thread="$3"
    local delay="$4"
    local padding_input="$5"
    local padding_output="$6"

    local machine_name="$(hostname)"
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local trace_dir="trace_${machine_name}_${timestamp}"
    local trace="${trace_dir}_copy1d_${workgroups}_wgp_${waves}_waves_delay${delay}_pad${padding_input}_${padding_output}"

    echo ""
    echo "========================================"
    echo "Profiling: copy_1d_dwordx4_static"
    echo "Workgroups: $workgroups, Waves: $waves"
    echo "Delay: $delay, Padding: [$padding_input, $padding_output]"
    echo "========================================"
    echo ""

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-activity 10 \
        -d \"$trace\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" \
        --num-workgroups \"$workgroups\" \
        --num-waves \"$waves\" \
        --num-elements-per-thread \"$elements_per_thread\" \
        --sched-delay-store \"$delay\" \
        --padding-bytes \"$padding_input\" \"$padding_output\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Parse command line argument for profiling
profile_kernel "3040" "5" "8" "3" "0" "0"
