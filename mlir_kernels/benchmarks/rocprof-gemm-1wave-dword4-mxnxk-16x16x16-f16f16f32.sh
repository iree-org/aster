#!/bin/bash

set -e

echo $(env)

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set: source the virtual environment"
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TEST_SCRIPT="mlir_kernels/test/test_gemm_1wave_dword4_mxnxk_16x16x16_f16f16f32.py"

profile_kernel() {
    local m="$1"
    local n="$2"
    local k="$3"
    local m_tile="${4:-16}"
    local n_tile="${5:-16}"
    local k_tile="${6:-16}"
    local mcpu="${7:-gfx942}"

    local machine_name="$(hostname)"
    local trace_dir="trace_${machine_name}"
    local trace="${trace_dir}_gemm_1wave_m${m}_n${n}_k${k}_mt${m_tile}_nt${n_tile}_kt${k_tile}"

    echo ""
    echo "========================================"
    echo "Profiling: gemm_1wave_dword4_mxnxk_16x16x16_f16f16f32"
    echo "m=$m, n=$n, k=$k"
    echo "Tile sizes: m_tile=$m_tile, n_tile=$n_tile, k_tile=$k_tile"
    echo "mcpu: $mcpu"
    echo "========================================"
    echo ""

    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-activity 10 \
        -d \"$trace\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" \
        --m \"$m\" \
        --n \"$n\" \
        --k \"$k\" \
        --m-tile \"$m_tile\" \
        --n-tile \"$n_tile\" \
        --k-tile \"$k_tile\" \
        --mcpu \"$mcpu\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Parse command line arguments for profiling
# Default: m=128, n=128, k=64, m_tile=16, n_tile=16, k_tile=16
profile_kernel "${1:-128}" "${2:-128}" "${3:-64}" "${4:-16}" "${5:-16}" "${6:-16}" "${7:-gfx942}"
