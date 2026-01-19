#!/bin/bash
# Profiling script for nanobench_global_load

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

check_venv

KERNEL_NAME="nanobench_global_load"
TEST_SCRIPT="${SCRIPT_DIR}/nanobench_global_load.py"
PERF_COUNTERS="TCC_HIT TCC_MISS TCC_READ TCC_WRITE"

profile_kernel() {
    local trace_prefix="$1"
    local num_kernel_runs="$2"
    local num_tiles="$3"
    local tile_size_bytes="$4"
    local tile_reuse_factor="$5"
    local dwordxbits="$6"
    local num_blocks="$7"
    local num_waves="$8"
    local flush_llc="$9"

    local params="num_kernel_runs=$num_kernel_runs, num_tiles=$num_tiles, tile_size_bytes=$tile_size_bytes, tile_reuse_factor=$tile_reuse_factor, dwordxbits=$dwordxbits, num_blocks=$num_blocks, num_waves=$num_waves, flush_llc=$flush_llc"
    print_profile_header "$KERNEL_NAME" "$params"

    local trace_dir
    trace_dir="$(make_trace_dir "${trace_prefix}_$KERNEL_NAME" "_runs${num_kernel_runs}_tiles${num_tiles}_size${tile_size_bytes}_reuse${tile_reuse_factor}_dwordxbits${dwordxbits}_blocks${num_blocks}_waves${num_waves}_flush${flush_llc}")"

    local flush_llc_flag=""
    if [ "$flush_llc" = "true" ]; then
        flush_llc_flag="--flush-llc"
    else
        flush_llc_flag="--no-flush-llc"
    fi

    # Note: use kernel-include-regex and kernel-iteration-range to profile only 
    # the kernel of interest on its 3rd iteration to remove icache effects.
    run_rocprof_att_filtered \
        "$trace_dir" \
        "$KERNEL_NAME" \
        "$PERF_COUNTERS" \
        "$KERNEL_NAME" \
        "[3-3]" \
        "$TEST_SCRIPT" \
        --num-kernel-runs "$num_kernel_runs" \
        --num-tiles "$num_tiles" \
        --tile-size-bytes "$tile_size_bytes" \
        --tile-reuse-factor "$tile_reuse_factor" \
        --dwordxbits "$dwordxbits" \
        --num-blocks "$num_blocks" \
        --num-waves "$num_waves" \
        $flush_llc_flag
}

for dwordxbits in 1 2 4 8; do
    case $dwordxbits in
        1) log2_dwordxbits=1 ;;
        2) log2_dwordxbits=2 ;;
        4) log2_dwordxbits=3 ;;
        8) log2_dwordxbits=4 ;;
    esac
    tile_size_bytes=$((log2_dwordxbits * 256)) # for latency benchmark

    #              trace_prefix      | num_runs  | num_tiles | tile_size_bytes          | tile_reuse | dwordxbits          | num_blocks  | num_waves | flush_llc
    # Latency benchmark L1: 1 tile of 64 x dword_size per wave, reuse factor 4
    profile_kernel "${1:-lat}"         "${2:-5}"   "${3:-1}"   "${4:-$tile_size_bytes}"   "${5:-8}"    "${6:-$dwordxbits}"   "${7:-1216}" "${8:-1}"    "${9:-false}"
    profile_kernel "${1:-lat}"         "${2:-5}"   "${3:-1}"   "${4:-$tile_size_bytes}"   "${5:-8}"    "${6:-$dwordxbits}"   "${7:-304}"  "${8:-4}"    "${9:-false}"
    # Bandwidth benchmark L1 (hot cache): 24 tiles of 256B per wave (24kB total vs 32kB L1), reuse factor 4
    profile_kernel "${1:-bw_hot_l1}"   "${2:-5}"   "${3:-24}"   "${4:-256}"               "${5:-8}"    "${6:-$dwordxbits}"   "${7:-1216}" "${8:-1}"    "${9:-false}"
    profile_kernel "${1:-bw_hot_l1}"   "${2:-5}"   "${3:-24}"   "${4:-256}"               "${5:-8}"    "${6:-$dwordxbits}"   "${7:-304}"  "${8:-4}"    "${9:-false}"
    # Bandwidth benchmark L2 (hot cache): 96 tiles of 256B per wave (96kB total vs 32kB L1), reuse factor 4
    profile_kernel "${1:-bw_hot_l2}"   "${2:-5}"   "${3:-96}"   "${4:-256}"               "${5:-8}"    "${6:-$dwordxbits}"   "${7:-1216}" "${8:-1}"    "${9:-false}"
    profile_kernel "${1:-bw_hot_l2}"   "${2:-5}"   "${3:-96}"   "${4:-256}"               "${5:-8}"    "${6:-$dwordxbits}"   "${7:-304}"  "${8:-4}"    "${9:-false}"
    # Bandwidth benchmark (cold cache): 120 tiles of 256B per wave (120kB total vs 32kB L1), reuse factor 1
    profile_kernel "${1:-bw_cold}"     "${2:-5}"   "${3:-120}"   "${4:-256}"              "${5:-1}"    "${6:-$dwordxbits}"   "${7:-1216}" "${8:-1}"    "${9:-true}"
    profile_kernel "${1:-bw_cold}"     "${2:-5}"   "${3:-120}"   "${4:-256}"              "${5:-1}"    "${6:-$dwordxbits}"   "${7:-304}"  "${8:-4}"    "${9:-true}"
done


# Note: run the following to check L2 cache hits / misses
# rocprofv3 -r --output-format csv --pmc TCC_HIT,TCC_MISS  -- python ./mlir_kernels/nanobenchmarks/nanobench_global_load.py   --num-blocks 304 --dwordxbits 1 --num-kernel-runs=1 --num-tiles 96 --tile-size-bytes 256 --tile-reuse-factor 1 

