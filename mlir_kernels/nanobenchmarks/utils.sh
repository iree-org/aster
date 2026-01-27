#!/bin/bash
# Common utilities for nanobenchmark profiling scripts.

set -e

# Check for virtual environment
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
        exit 1
    fi
}

# Get standard paths
get_python_bin() {
    echo "${VIRTUAL_ENV}/bin/python"
}

get_script_dir() {
    echo "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}

get_machine_name() {
    hostname
}

# Print profiling header
print_profile_header() {
    local kernel_name="$1"
    shift
    local params="$@"

    echo ""
    echo "========================================"
    echo "Profiling: $kernel_name"
    echo "$params"
    echo "========================================"
    echo ""
}

# Run rocprofv3 with ATT tracing
# Args: trace_dir kernel_name python_script [script_args...]
run_rocprof_att() {
    local trace_dir="$1"
    local kernel_name="$2"
    local python_script="$3"
    shift 3
    local script_args="$@"

    local python_bin
    python_bin="$(get_python_bin)"

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        -d \"$trace_dir\" \
        -- \
        \"$python_bin\" \"$python_script\" $script_args"

    echo "Command: $cmd"
    eval "$cmd"
}

# Run rocprofv3 with ATT tracing and performance counters
# Args: trace_dir kernel_name perf_counters python_script [script_args...]
run_rocprof_att_perf() {
    local trace_dir="$1"
    local kernel_name="$2"
    local perf_counters="$3"
    local python_script="$4"
    shift 4
    local script_args="$@"

    local python_bin
    python_bin="$(get_python_bin)"

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        --att-perfcounters \"$perf_counters\" \
        -d \"$trace_dir\" \
        -- \
        \"$python_bin\" \"$python_script\" $script_args"

    echo "Command: $cmd"
    eval "$cmd"
}

# Run rocprofv3 with ATT tracing, perf counters, and kernel filtering
# Args: trace_dir kernel_name perf_counters kernel_regex iter_range python_script [script_args...]
run_rocprof_att_filtered() {
    local trace_dir="$1"
    local kernel_name="$2"
    local perf_counters="$3"
    local kernel_regex="$4"
    local iter_range="$5"
    local python_script="$6"
    shift 6
    local script_args="$@"

    local python_bin
    python_bin="$(get_python_bin)"

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        --att-perfcounters \"$perf_counters\" \
        -d \"$trace_dir\" \
        --kernel-include-regex \"$kernel_regex\" \
        --kernel-iteration-range \"$iter_range\" \
        -- \
        \"$python_bin\" \"$python_script\" $script_args"

    echo "Command: $cmd"
    eval "$cmd"
}

# Generate trace directory name
# Args: kernel_name param_suffix
make_trace_dir() {
    local kernel_name="$1"
    local param_suffix="$2"
    local machine_name
    machine_name="$(get_machine_name)"
    echo "trace_${machine_name}_${kernel_name}${param_suffix}"
}
