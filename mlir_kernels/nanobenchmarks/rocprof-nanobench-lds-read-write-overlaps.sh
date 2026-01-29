#!/bin/bash
# Profiling script for nanobench_lds_read_write_overlaps
# Tests global_load -> ds_write -> ds_read pipeline with explicit waits

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

check_venv

KERNEL_NAME="nanobench_lds_read_write_overlaps"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_lds_read_write_overlaps.py"
# Counters for global loads (VM) and LDS operations
PERF_COUNTERS="SQ_INSTS_VMEM SQ_INSTS_LDS SQ_WAIT_INST_VMEM SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT"

profile_kernel() {
    local num_iters="$1"
    local num_kernel_runs="$2"
    local num_blocks="$3"

    local params="num_iters=$num_iters, num_kernel_runs=$num_kernel_runs, num_blocks=$num_blocks"
    print_profile_header "$KERNEL_NAME" "$params"

    local trace_dir
    trace_dir=$(make_trace_dir "$KERNEL_NAME" "_iters${num_iters}_runs${num_kernel_runs}_blocks${num_blocks}")

    run_rocprof_att_perf \
        "$trace_dir" \
        "$KERNEL_NAME" \
        "$PERF_COUNTERS" \
        "$TEST_SCRIPT" \
        --num-iters "$num_iters" \
        --num-kernel-runs "$num_kernel_runs" \
        --num-blocks "$num_blocks"
}

# Default: num_iters=3, num_kernel_runs=10, num_blocks=304
profile_kernel "${1:-3}" "${2:-10}" "${3:-304}"
