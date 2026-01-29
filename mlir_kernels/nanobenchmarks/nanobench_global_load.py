#!/usr/bin/env python3
"""Nanobenchmark for @global_load_wave with configurable multiple tiles/waves/workgroups per CU."""

import argparse
import os

import numpy as np

from utils import (
    MCPU,
    NANOBENCH_PASS_PIPELINE,
    WAVEFRONT_SIZE,
    NanobenchConfig,
    add_common_args,
    compile_kernel,
    run_kernel,
    print_timing_stats,
)

KERNEL_NAME = "nanobench_global_load"


def main():
    parser = argparse.ArgumentParser(
        description="Nanobenchmark for global load with configurable multiple tiles/waves/workgroups per CU"
    )
    add_common_args(parser)
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=4,
        help="Number of tiles to load (default: 4)",
    )
    parser.add_argument(
        "--tile-size-bytes",
        type=int,
        default=384,
        help="Size of each tile in bytes (default: 384)",
    )
    parser.add_argument(
        "--tile-reuse-factor",
        type=int,
        default=1,
        help="Tile reuse factor (default: 1)",
    )
    parser.add_argument(
        "--flush-llc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush LLC before each kernel run (default: True)",
    )
    parser.add_argument(
        "--dwordxbits",
        type=int,
        default=15,
        help="DWORDX bits: 1=dword, 2=dwordx2, 4=dwordx3, 8=dwordx4 (default: 15=all)",
    )
    # Don't support num-iters, this is all about tile-reuse-factor.
    # Remove --num-iters from parser (it was added by add_common_args)
    # argparse converts --num-iters to dest="num_iters"
    parser._actions = [
        action for action in parser._actions if action.dest != "num_iters"
    ]
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(script_dir, "nanobench_global_load.mlir")

    num_threads = args.num_waves * WAVEFRONT_SIZE
    num_bytes = args.num_blocks * args.num_waves * args.num_tiles * args.tile_size_bytes

    def preprocess(x, dwordxbits=args.dwordxbits):
        x = x.replace("{{NUM_TILES}}", str(args.num_tiles))
        x = x.replace("{{TILE_SIZE_BYTES}}", str(args.tile_size_bytes))
        x = x.replace("{{TILE_REUSE_FACTOR}}", str(args.tile_reuse_factor))
        x = x.replace("{{NUM_WAVES}}", str(args.num_waves))
        x = x.replace("{{NUM_THREADS}}", str(num_threads))
        x = x.replace("{{NUM_BLOCKS}}", str(args.num_blocks))
        x = x.replace("{{DWORDXBITS}}", str(dwordxbits))
        return x

    input_buffer = np.random.randn(num_bytes).astype(np.uint8)

    config = NanobenchConfig(
        kernel_name=KERNEL_NAME,
        mlir_file=mlir_file,
        pass_pipeline=NANOBENCH_PASS_PIPELINE,
        description="global load benchmark",
        num_blocks=args.num_blocks,
        num_threads=num_threads,
        num_iters=args.tile_reuse_factor,
        num_kernel_runs=args.num_kernel_runs,
        input_buffers=[input_buffer],
        flush_llc=args.flush_llc,
        print_asm=args.print_asm,
    )

    hsaco_path, _ = compile_kernel(config, preprocess)
    iteration_times_ns = run_kernel(config, hsaco_path)

    if iteration_times_ns is not None:
        variants = [
            f"dword{'x'+str(i) if i > 1 else ''}"
            for i in [1, 2, 3, 4]
            if args.dwordxbits & (1 << (i - 1))
        ]
        print_timing_stats(
            iteration_times_ns,
            args.num_kernel_runs,
            extra_info={
                "variants": "+".join(variants),
                "num_bytes": f"{num_bytes/1e6} MB",
            },
        )


if __name__ == "__main__":
    main()
