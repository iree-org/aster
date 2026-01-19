#!/bin/bash
# Profiling script for nanobench_global_load

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

check_venv

KERNEL_NAME="nanobench_global_load"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_global_load.py"
PERF_COUNTERS="TCC_HIT TCC_MISS TCC_READ TCC_WRITE"

profile_kernel() {
    local num_kernel_runs="$1"
    local num_iters="$2"
    local num_tiles="$3"
    local tile_reuse_factor="$4"
    local dwordxbits="$5"
    local num_blocks="$6"
    local num_waves="$7"

    local params="num_kernel_runs=$num_kernel_runs, num_iters=$num_iters, num_tiles=$num_tiles, tile_reuse_factor=$tile_reuse_factor, dwordxbits=$dwordxbits, num_blocks=$num_blocks, num_waves=$num_waves"
    print_profile_header "$KERNEL_NAME" "$params"

    local trace_dir
    trace_dir=$(make_trace_dir "$KERNEL_NAME" "_iters${num_iters}_runs${num_kernel_runs}_tiles${num_tiles}_reuse${tile_reuse_factor}_dword${dwordxbits}_blocks${num_blocks}_waves${num_waves}")

    # Note: use kernel-include-regex and kernel-iteration-range to profile only 
    # the kernel of interest on its 3rd iteration to remove icache effects.
    run_rocprof_att_filtered \
        "$trace_dir" \
        "$KERNEL_NAME" \
        "$PERF_COUNTERS" \
        "$KERNEL_NAME" \
        "[3-3]" \
        "$TEST_SCRIPT" \
        --num-iters "$num_iters" \
        --num-kernel-runs "$num_kernel_runs" \
        --num-tiles "$num_tiles" \
        --tile-reuse-factor "$tile_reuse_factor" \
        --dwordxbits "$dwordxbits" \
        --num-blocks "$num_blocks" \
        --num-waves "$num_waves"
}

# Note: a tile is 256B so with dword it is fully loaded in a single wave load.
# So 4 tiles is the atomic unit of load for dwordxbits4.

# Latency benchmark: 5 iterations, 4 tiles of 256B per wave, reuse factor 1
#              num_runs num_iters num_tiles tile_reuse dword_size num_blocks num_waves
profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-1}" "${5:-15}" "${6:-1216}" "${7:-1}"
profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-1}" "${5:-15}" "${6:-304}" "${7:-4}"

# Bandwidth benchmark (hot cache): 5 iterations, 4 tiles of 256B per wave, reuse factor 16
#              num_runs num_iters num_tiles tile_reuse dword_size num_blocks num_waves
profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-16}" "${5:-15}" "${6:-1216}" "${7:-1}"
profile_kernel "${1:-5}" "${2:-5}" "${3:-4}" "${4:-16}" "${5:-15}" "${6:-304}" "${7:-4}"

# Bandwidth benchmark (cold cache): 1 iteration, 128 tiles of 256B per wave, reuse factor 1
for dwordxbits in 1 2 4 8; do
    #              num_runs num_iters num_tiles tile_reuse dword_size num_blocks num_waves
    profile_kernel "${1:-5}" "${2:-1}" "${3:-128}" "${4:-1}" "${5:-$dwordxbits}" "${6:-1216}" "${7:-1}"
    profile_kernel "${1:-5}" "${2:-1}" "${3:-128}" "${4:-1}" "${5:-$dwordxbits}" "${6:-304}" "${7:-4}"
done
