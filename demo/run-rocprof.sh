#!/bin/bash

set -e

echo $(env)

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TEST_SCRIPT="demo/run.py"

# Parse command line arguments
# First argument is MLIR file (default: demo/add_10.mlir)
MLIR_FILE="${1:-demo/add_10.mlir}"
shift  # Remove first argument, remaining args are passed through to run.py
EXTRA_ARGS="$@"  # Capture remaining arguments to pass to run.py

profile_kernel() {
    local machine_name="$(hostname)"
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local trace_dir="trace_${machine_name}_${timestamp}"
    local kernel_name=$(basename "$MLIR_FILE" .mlir)
    local trace="${trace_dir}_${kernel_name}"

    echo ""
    echo "========================================"
    echo "Profiling: $kernel_name"
    echo "========================================"
    echo ""

    # TODO: ideally we'd want to use these arguments to profile the minimal
    # amount of asm possible for the kernel, but I don't find the right
    # incantations on CDNA.
    # --att-shader-engine-mask 0xF \
    # --att-target-cu 0 \
    # --att-simd-select 0 \
    #
    # Note: we profile the 3rd iteration to remove icache effects that can
    # skew the trace on the first iteration.
    local cmd="/usr/bin/rocprofv3 \
        --kernel-iteration-range 3 \
        --kernel-include-regex \".*kernel.*\" \
        --att \
        --att-activity 10 \
        -d \"$trace\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" --mlir-file \"$MLIR_FILE\" --num-iterations 5 $EXTRA_ARGS
    "
    echo "Command: $cmd"
    eval "$cmd"
}


# Parse command line argument for profiling
profile_kernel
