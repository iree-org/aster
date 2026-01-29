#!/usr/bin/env python3
"""Nanobenchmark to stress the wait optimization pass for peak LDS usage."""

import argparse
import os

import numpy as np

from utils import (
    NANOBENCH_PASS_PIPELINE,
    WAVEFRONT_SIZE,
    NanobenchConfig,
    add_common_args,
    compile_kernel,
    run_kernel,
    print_per_call_stats,
)

KERNEL_NAME = "nanobench_lds_read_write_overlaps"

# Default parameters
DEFAULT_NUM_TILES_I = 2  # Number of 16x16 tiles in M direction
DEFAULT_NUM_TILES_J = 4  # Number of 16x16 tiles in N direction
DEFAULT_GLOBAL_STRIDE_BYTES = 64 * 2  # 64 f16 elements = 128 bytes
DEFAULT_LDS_STRIDE_BYTES = 64 * 2  # 64 f16 elements = 128 bytes


def compute_lds_size(num_tiles_i: int, num_tiles_j: int, lds_stride_bytes: int) -> int:
    """Compute required LDS size for the given tile configuration."""
    rows = num_tiles_i * 16
    return rows * lds_stride_bytes


def compute_buffer_size(
    num_tiles_i: int, num_tiles_j: int, global_stride_bytes: int
) -> int:
    """Compute required input buffer size in bytes."""
    rows = num_tiles_i * 16
    cols = global_stride_bytes
    return rows * cols


def main():
    parser = argparse.ArgumentParser(
        description="Nanobenchmark for multi-tile overlapped operations"
    )
    add_common_args(parser)
    # Override num_iters default to 3 for this nanobenchmark
    for action in parser._actions:
        if action.dest == "num_iters":
            action.default = 3
            action.help = "Number of inner loop iterations (default: 3)"
            break
    parser.add_argument(
        "--num-tiles-i",
        type=int,
        default=DEFAULT_NUM_TILES_I,
        help=f"Number of 16x16 tiles in M direction (default: {DEFAULT_NUM_TILES_I})",
    )
    parser.add_argument(
        "--num-tiles-j",
        type=int,
        default=DEFAULT_NUM_TILES_J,
        help=f"Number of 16x16 tiles in N direction (default: {DEFAULT_NUM_TILES_J})",
    )
    parser.add_argument(
        "--global-stride-bytes",
        type=int,
        default=DEFAULT_GLOBAL_STRIDE_BYTES,
        help=f"Global memory stride in bytes (default: {DEFAULT_GLOBAL_STRIDE_BYTES})",
    )
    parser.add_argument(
        "--lds-stride-bytes",
        type=int,
        default=DEFAULT_LDS_STRIDE_BYTES,
        help=f"LDS stride in bytes (default: {DEFAULT_LDS_STRIDE_BYTES})",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(script_dir, "nanobench_lds_read_write_overlaps.mlir")

    lds_size = compute_lds_size(
        args.num_tiles_i, args.num_tiles_j, args.lds_stride_bytes
    )
    buffer_size = compute_buffer_size(
        args.num_tiles_i, args.num_tiles_j, args.global_stride_bytes
    )
    num_elements = buffer_size // 2  # f16 = 2 bytes
    num_tiles = args.num_tiles_i * args.num_tiles_j

    def preprocess(x):
        x = x.replace("{{NUM_ITERS}}", str(args.num_iters))
        x = x.replace("{{NUM_TILES_I}}", str(args.num_tiles_i))
        x = x.replace("{{NUM_TILES_J}}", str(args.num_tiles_j))
        x = x.replace("{{NUM_TILES}}", str(num_tiles))
        x = x.replace("{{GLOBAL_STRIDE_BYTES}}", str(args.global_stride_bytes))
        x = x.replace("{{LDS_STRIDE_BYTES}}", str(args.lds_stride_bytes))
        x = x.replace("{{LDS_SIZE}}", str(lds_size))
        x = x.replace("{{NUM_THREADS}}", str(WAVEFRONT_SIZE))
        x = x.replace("{{NUM_BLOCKS}}", str(args.num_blocks))
        return x

    # Create input buffer with data
    input_data = np.arange(num_elements, dtype=np.float16)
    input_buffers = [input_data]

    config = NanobenchConfig(
        kernel_name=KERNEL_NAME,
        mlir_file=mlir_file,
        pass_pipeline=NANOBENCH_PASS_PIPELINE,
        description="Multi-tile overlapped global_load -> ds_write -> ds_read nanobenchmark",
        num_blocks=args.num_blocks,
        num_threads=WAVEFRONT_SIZE,
        num_iters=args.num_iters,
        num_kernel_runs=args.num_kernel_runs,
        input_buffers=input_buffers,
        print_asm=args.print_asm,
        print_timings=False,
        print_ir_after_all=False,
    )

    # Compile and get assembly to analyze wait instructions
    hsaco_path, asm = compile_kernel(config, preprocess)

    # Count s_waitcnt instructions in the generated assembly
    waitcnt_count = asm.count("s_waitcnt")
    vmcnt_count = asm.count("vmcnt")
    lgkmcnt_count = asm.count("lgkmcnt")

    print(f"\nWait instruction analysis:")
    print(f"  Total s_waitcnt: {waitcnt_count}")
    print(f"  vmcnt references: {vmcnt_count}")
    print(f"  lgkmcnt references: {lgkmcnt_count}")

    # Run the kernel
    iteration_times_ns = run_kernel(config, hsaco_path)

    if iteration_times_ns is not None:
        calls_per_iter = num_tiles
        print_per_call_stats(
            iteration_times_ns,
            args.num_kernel_runs,
            calls_per_iter,
            args.num_iters,
        )


if __name__ == "__main__":
    main()
