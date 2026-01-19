#!/usr/bin/env python3
"""Nanobenchmark for @lds_read_swizzled_wave_16x16xf16_fragment_wait."""

import argparse
import os

from utils import (
    WAVEFRONT_SIZE,
    NanobenchConfig,
    add_common_args,
    run_nanobenchmark,
    print_per_call_stats,
)

KERNEL_NAME = "nanobench_lds_read_swizzled_wave_16x16xf16"
LDS_SIZE = 16384
CALLS_PER_ITER = 4 * 8  # II=4, JJ=8


def main():
    parser = argparse.ArgumentParser(
        description="Nanobenchmark for LDS read swizzled wave 16x16xf16 fragment"
    )
    add_common_args(parser)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(script_dir, "nanobench_lds_read_swizzled_wave_16x16xf16.mlir")

    def preprocess(x):
        x = x.replace("{{NUM_ITERS}}", str(args.num_iters))
        x = x.replace("{{LDS_SIZE}}", str(LDS_SIZE))
        x = x.replace("{{NUM_THREADS}}", str(WAVEFRONT_SIZE))
        x = x.replace("{{NUM_BLOCKS}}", str(args.num_blocks))
        return x

    config = NanobenchConfig(
        kernel_name=KERNEL_NAME,
        mlir_file=mlir_file,
        description="LDS read swizzled wave 16x16xf16 benchmark",
        num_blocks=args.num_blocks,
        num_threads=WAVEFRONT_SIZE,
        num_iters=args.num_iters,
        num_kernel_runs=args.num_kernel_runs,
        print_asm=args.print_asm,
    )

    def stats_fn(iteration_times_ns):
        print_per_call_stats(
            iteration_times_ns,
            args.num_kernel_runs,
            CALLS_PER_ITER,
            args.num_iters,
        )

    run_nanobenchmark(config, preprocess, stats_fn=stats_fn)


if __name__ == "__main__":
    main()
