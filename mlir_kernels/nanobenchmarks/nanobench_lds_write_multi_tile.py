#!/usr/bin/env python3
"""Nanobenchmark for @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait."""

import argparse
import os

from utils import (
    NANOBENCH_PASS_PIPELINE,
    WAVEFRONT_SIZE,
    NanobenchConfig,
    add_common_args,
    run_nanobenchmark,
    print_per_call_stats,
)

KERNEL_NAME = "nanobench_lds_write_multi_tile"
LDS_SIZE = 16384
CALLS_PER_ITER = 4  # (II/NT_I) * (JJ/NT_J) = 2*2 = 4


def main():
    parser = argparse.ArgumentParser(
        description="Nanobenchmark for LDS write multi-tile coalesced"
    )
    add_common_args(parser)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(script_dir, "nanobench_lds_write_multi_tile.mlir")

    def preprocess(x):
        x = x.replace("{{NUM_ITERS}}", str(args.num_iters))
        x = x.replace("{{LDS_SIZE}}", str(LDS_SIZE))
        x = x.replace("{{NUM_THREADS}}", str(WAVEFRONT_SIZE))
        x = x.replace("{{NUM_BLOCKS}}", str(args.num_blocks))
        return x

    config = NanobenchConfig(
        kernel_name=KERNEL_NAME,
        mlir_file=mlir_file,
        pass_pipeline=NANOBENCH_PASS_PIPELINE,
        description="LDS write multi-tile benchmark",
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
