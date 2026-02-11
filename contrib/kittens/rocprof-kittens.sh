#!/bin/bash
# Profile kittens kernels with rocprofv3 ATT tracing.
#
# Profiles all kernels by default. Comment out lines at the bottom to select.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/../../mlir_kernels/nanobenchmarks/utils.sh"
check_venv

PYTHON_BIN="$(get_python_bin)"
TEST_SCRIPT="${SCRIPT_DIR}/test/test_lib.py"
NUM_BLOCKS="${1:-304}"
PERF_COUNTERS="SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_INSTS_VMEM SQ_WAIT_INST_VMEM TCC_HIT TCC_MISS"

profile_kernel() {
    local kernel="$1"
    local trace_dir
    trace_dir="$(make_trace_dir "kittens_${kernel}" "_blocks${NUM_BLOCKS}")"
    echo "Profiling: $kernel (${NUM_BLOCKS} blocks) -> $trace_dir"
    /usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        --att-perfcounters "$PERF_COUNTERS" \
        --kernel-include-regex "$kernel" \
        --kernel-iteration-range "[3-5]" \
        -d "$trace_dir" \
        -- \
        "$PYTHON_BIN" "$TEST_SCRIPT" \
        --num-blocks "$NUM_BLOCKS" \
        --num-iterations 5
}

# profile_kernel test_zero_C
# profile_kernel test_load_store_A
# profile_kernel test_mfma
profile_kernel gemm_16x16x128
