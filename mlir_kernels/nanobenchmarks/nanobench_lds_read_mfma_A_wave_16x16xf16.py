#!/usr/bin/env python3
"""Nanobenchmark for @lds_read_A_wave_16x16xf16_fragment_wait."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from aster import ir, utils
from integration_test.test_utils import (
    compile_mlir_file_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)
from mlir_kernels.common import get_library_paths, NANOBENCH_PASS_PIPELINE

KERNEL_NAME = "nanobench_lds_read_mfma_A_wave_16x16xf16"
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def main():
    parser = argparse.ArgumentParser(
        description="Nanobenchmark for LDS read A wave 16x16xf16 fragment"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of outer loop iterations (default: 10)",
    )
    parser.add_argument(
        "--num-kernel-runs",
        type=int,
        default=100,
        help="Number of kernel invocations for timing (default: 100)",
    )
    parser.add_argument(
        "--num-cus",
        type=int,
        default=1,
        help="Number of CUs to use (default: 1)",
    )
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Print generated assembly",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(script_dir, "nanobench_lds_read_mfma_A_wave_16x16xf16.mlir")
    library_paths = get_library_paths()

    def preprocess(x):
        x = x.replace("{{NUM_ITERS}}", str(args.num_iters))
        x = x.replace("{{LDS_SIZE}}", str(16384))
        x = x.replace("{{NUM_THREADS}}", str(64))
        x = x.replace("{{NUM_BLOCKS}}", str(args.num_cus))
        return x

    with ir.Context() as ctx:
        asm_complete, _ = compile_mlir_file_to_asm(
            mlir_file,
            KERNEL_NAME,
            NANOBENCH_PASS_PIPELINE,
            ctx,
            preprocess=preprocess,
            library_paths=library_paths,
        )

        if args.print_asm:
            print(asm_complete)

        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        print(f"Compiled successfully. HSACO: {hsaco_path}")
        print(f"Config: {args.num_iters} inner iterations, {args.num_kernel_runs} kernel runs")

        if not utils.system_has_mcpu(mcpu=MCPU):
            print(f"GPU {MCPU} not available, stopping after cross-compilation")
            return

        with hsaco_file(hsaco_path):
            iteration_times_ns = execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=KERNEL_NAME,
                input_args=[],
                output_args=[],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=(args.num_cus, 1, 1),
                block_dim=(WAVEFRONT_SIZE, 1, 1),
                verify_fn=None,
                num_iterations=args.num_kernel_runs,
            )

            # Stats
            import numpy as np
            times_us = np.array(iteration_times_ns) / 1000.0
            # II=4, JJ=8 means 32 calls per iteration
            calls_per_iter = 4 * 8  # II * JJ = 32
            total_calls = args.num_iters * calls_per_iter

            print(f"\nTiming results ({args.num_kernel_runs} runs):")
            print(f"  Mean: {np.mean(times_us):.2f} us")
            print(f"  Min:  {np.min(times_us):.2f} us")
            print(f"  Max:  {np.max(times_us):.2f} us")
            print(f"  Std:  {np.std(times_us):.2f} us")
            print(f"\nPer-call estimate: {np.mean(times_us) * 1000 / total_calls:.2f} ns "
                  f"({total_calls} calls per kernel)")


if __name__ == "__main__":
    main()
