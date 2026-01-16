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
    local num_kernel_runs="$1"
    local num_iters="$2"
    local num_tiles="$3"
    local tile_reuse_factor="$4"
    local dwordx="$5"
    local num_waves="$6"
    local num_cus="$7"

    local machine_name="$(hostname)"
    local trace="trace_${machine_name}_nanobench_global_load${num_iters}_runs${num_kernel_runs}_num_tiles${num_tiles}_tile_reuse_factor${tile_reuse_factor}_dwordx${dwordx}_num_waves${num_waves}_num_cus${num_cus}"

    echo ""
    echo "========================================"
    echo "Profiling: nanobench_global_load"
    echo "num_kernel_runs=$num_kernel_runs, num_iters=$num_iters"
    echo "num_tiles=$num_tiles, tile_reuse_factor=$tile_reuse_factor, dwordx=$dwordx"
    echo "num_waves=$num_cus, num_cus=$num_waves"
    echo "========================================"
    echo ""

    # Note: use kernel-include-regex and kernel-iteration-range to profile only 
    # the kernel of interest on its 3rd iteration to remove icache effects that
    # can skew the trace.
    local cmd="/usr/bin/rocprofv3 \
        --att \
        --att-perfcounter-ctrl 10 \
        --att-perfcounters \"TCC_HIT TCC_MISS TCC_READ TCC_WRITE\" \
        -d \"$trace\" \
        --kernel-include-regex \"nanobench_global_load\" \
        --kernel-iteration-range \"[3-3]\" \
        -- \
        \"$PYTHON_BIN\" \"$TEST_SCRIPT\" \
        --num-iters \"$num_iters\" \
        --num-kernel-runs \"$num_kernel_runs\" \
        --num-tiles \"$num_tiles\" \
        --tile-reuse-factor \"$tile_reuse_factor\" \
        --dwordx \"$dwordx\" \
        --num-waves \"$num_waves\" \
        --num-cus \"$num_cus\""
    echo "Command: $cmd"
    eval "$cmd"
}

# Note: a tile is 256B so with dword it is fully loaded in a single wave load.
# So 4 tiles is the atomic unit of load for dwordx4.
for dwordx in 1 2 3 4; do
    # Latency benchmark: 5 iterations, 4 tiles of 256B, reuse factor 1
    # 1216 workgroups 1 wave
    #              num_runs num_iters num_tiles tile_reuse    dword_size   num_waves num_cus
    profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-1}" "${5:-$dwordx}" "${6:-1}" "${7:-1216}"
    # 304 workgroups 4 waves
    profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-1}" "${5:-$dwordx}" "${6:-4}" "${7:-304}"

    # Bandwidth benchmark (cold cache): 1 iteration, 128 tiles of 256B, reuse factor 1
    #              num_runs num_iters num_tiles tile_reuse    dword_size   num_waves num_cus
    # 1216 workgroups 1 wave
    profile_kernel "${1:-5}" "${2:-1}" "${3:-128}" "${4:-1}" "${5:-$dwordx}" "${6:-1}" "${7:-1216}"
    # 304 workgroups 4 waves
    profile_kernel "${1:-5}" "${2:-1}" "${3:-128}" "${4:-1}" "${5:-$dwordx}" "${6:-4}" "${7:-304}"

    # Bandwidth benchmark (hot cache): 5 iterations, 4 tiles of 256B, reuse factor 16
    #              num_runs num_iters num_tiles tile_reuse    dword_size   num_waves num_cus
    # 1216 workgroups 1 wave
    profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-16}" "${5:-$dwordx}" "${6:-1}" "${7:-1216}"
    # 304 workgroups 4 waves
    profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-16}" "${5:-$dwordx}" "${6:-4}" "${7:-304}"
done
