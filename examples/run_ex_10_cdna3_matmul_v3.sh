#!/bin/bash

set -e

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
EXAMPLE="ex_10_cdna3_matmul_v3.py"

# Helper to generate complete output filename from extra args
generate_output_filename() {
    local base_name="$(basename "$EXAMPLE" .py)"
    local suffix=""
    for arg in "$@"; do
        case "$arg" in
            --m-regs=*)
                suffix="${suffix}_m${arg#*=}"
                ;;
            --n-regs=*)
                suffix="${suffix}_n${arg#*=}"
                ;;
            --k-regs=*)
                suffix="${suffix}_k${arg#*=}"
                ;;
            --num-vgprs=*)
                suffix="${suffix}_vgpr${arg#*=}"
                ;;
            --num-agprs=*)
                suffix="${suffix}_agpr${arg#*=}"
                ;;
            --prefetch-amount=*)
                suffix="${suffix}_pf${arg#*=}"
                ;;
            --iteration-step-size=*)
                suffix="${suffix}_iss${arg#*=}"
                ;;
        esac
    done
    echo "${base_name}${suffix}.hsaco"
}

build_kernel() {
    local output="$1"
    shift 1
    local extra_args="$@"

    echo ""
    echo "========================================"
    echo "Building: $EXAMPLE -> $output"
    echo "Iterations: $iterations"
    echo "Arguments: $extra_args"
    echo "========================================"
    echo ""

    local cmd="$PYTHON_BIN \"$EXAMPLE\" --output=\"$output\" --mcpu gfx942 $extra_args"
    echo "Command: $cmd"
    eval "$cmd"

    echo ""
    echo "Built: $output"
    echo ""
}

profile_kernel() {
    local workgroups="$1"
    local wavefronts="$2"
    local hsaco="$3"

    # Check if rocprofv3 is available
    if ! command -v rocprofv3 &> /dev/null; then
        echo "Warning: rocprofv3 not found. Skipping profiling."
        return 0
    fi

    local example_name="$(basename "$EXAMPLE" .py)"
    local machine_name="$(hostname)"
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local trace_dir="trace_${machine_name}_${timestamp}"
    local hsaco_name="$(basename "$hsaco" .hsaco)"
    local trace="${trace_dir}_${hsaco_name}_${workgroups}_wgp_${wavefronts}_wvf"

    echo ""
    echo "========================================"
    echo "Profiling: $EXAMPLE with $hsaco"
    echo "Workgroups: $workgroups, Wavefronts: $wavefronts"
    echo "========================================"
    echo ""

    local cmd="rocprofv3 --kernel-iteration-range 3 --kernel-include-regex \"kernel\" --att --att-activity 10 -d \"$trace\" -- $PYTHON_BIN \"ex_10_cdna3_matmul_exec.py\" --input=\"$hsaco\" --num-workgroups \"$workgroups\" --num-wavefronts \"$wavefronts\" --num-iterations 5"
    echo "Command: $cmd"
    eval "$cmd"
}

# Array to store profile jobs
profile_jobs=()

# Parse command line argument for which config to run
CONFIG="${1:-all}"
PROFILE="${2:-no}"  # Optional second argument to enable profiling

# Configuration 4x4x4
if [[ "$CONFIG" == "4x4x4" ]] || [[ "$CONFIG" == "all" ]]; then
    echo "Building 4x4x4 configuration (4x4x4)..."
    extra_args="--m-regs=4 --n-regs=4 --k-regs=4 --iteration-step-size=4 --prefetch-amount=12 --num-vgprs=120 --num-agprs=68"
    output="$(generate_output_filename $extra_args)"
    build_kernel "$output" $extra_args
    profile_jobs+=("profile_kernel \"256\" \"8\" \"$output\"")
fi

if [[ "$CONFIG" == "6x8x4" ]] || [[ "$CONFIG" == "all" ]]; then
    echo "Building 6x8x4 configuration (6x8x4)..."
    extra_args="--m-regs=6 --n-regs=8 --k-regs=4 --iteration-step-size=3 --prefetch-amount=6 --num-vgprs=120 --num-agprs=200"
    output="$(generate_output_filename $extra_args)"
    build_kernel "$output" $extra_args
    profile_jobs+=("profile_kernel \"256\" \"8\" \"$output\"")
fi

echo ""
echo "========================================"
echo "All builds completed!"
echo "========================================"

# Run all profile jobs if profiling is enabled
if [[ "$PROFILE" == "yes" ]] && [ ${#profile_jobs[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Starting profiling (${#profile_jobs[@]} jobs)..."
    echo "========================================"
    echo ""
    for job in "${profile_jobs[@]}"; do
        eval "$job"
    done
    echo ""
    echo "========================================"
    echo "All profiling completed!"
    echo "========================================"
fi
