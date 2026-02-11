#!/usr/bin/env python3
"""Common utilities for nanobenchmarks."""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../test"))

from aster import ir, utils
from integration.test_utils import (
    compile_mlir_file_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)
from integration.flush_llc import FlushLLC
from mlir_kernels.common import get_library_paths
from aster.pass_pipelines import NANOBENCH_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64


@dataclass
class NanobenchConfig:
    """Configuration for a nanobenchmark."""

    kernel_name: str
    mlir_file: str
    pass_pipeline: str
    description: str
    # Grid/block dimensions
    num_blocks: int = 1
    num_threads: int = WAVEFRONT_SIZE
    # Iteration counts
    num_iters: int = 10
    num_kernel_runs: int = 100
    # Optional input/output buffers
    input_buffers: list = field(default_factory=list)
    output_buffers: list = field(default_factory=list)
    # Optional flush LLC
    flush_llc: bool = False
    # Print options
    print_asm: bool = False
    print_timings: bool = False
    print_ir_after_all: bool = False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all nanobenchmarks."""
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of inner loop iterations (default: 10)",
    )
    parser.add_argument(
        "--num-kernel-runs",
        type=int,
        default=100,
        help="Number of kernel invocations for timing (default: 100)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of blocks/CUs to use (default: 1)",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        default=1,
        help="Number of waves per block (default: 1)",
    )
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Print generated assembly",
    )


def compile_kernel(
    config: NanobenchConfig,
    preprocess: Callable[[str], str],
) -> tuple[str, str]:
    """Compile MLIR to HSACO.

    Returns (hsaco_path, asm) or raises RuntimeError on failure.
    """
    library_paths = get_library_paths()

    with ir.Context() as ctx:
        asm_complete, _ = compile_mlir_file_to_asm(
            config.mlir_file,
            config.kernel_name,
            config.pass_pipeline,
            ctx,
            preprocess=preprocess,
            library_paths=library_paths,
            print_timings=config.print_timings,
            print_ir_after_all=config.print_ir_after_all,
        )

        if config.print_asm:
            print(asm_complete)

        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        return hsaco_path, asm_complete


def run_kernel(
    config: NanobenchConfig,
    hsaco_path: str,
    verify_fn: Optional[Callable] = None,
) -> Optional[list[int]]:
    """Run the kernel and return iteration times in nanoseconds.

    Returns None if GPU is not available.
    """
    print(f"Compiled successfully. HSACO: {hsaco_path}")
    print(
        f"Config: {config.num_iters} inner iterations, {config.num_kernel_runs} kernel runs, "
        f"{config.num_blocks} blocks, {config.num_threads} threads/block"
    )

    if not utils.system_has_mcpu(mcpu=MCPU):
        print(f"GPU {MCPU} not available, stopping after cross-compilation")
        return None

    flush_llc = FlushLLC(mcpu=MCPU) if config.flush_llc else None

    with hsaco_file(hsaco_path):
        iteration_times_ns = execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name=config.kernel_name,
            input_args=config.input_buffers,
            output_args=config.output_buffers,
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            grid_dim=(config.num_blocks, 1, 1),
            block_dim=(config.num_threads, 1, 1),
            verify_fn=verify_fn,
            num_iterations=config.num_kernel_runs,
            flush_llc=flush_llc,
        )
        return iteration_times_ns


def print_timing_stats(
    iteration_times_ns: list[int],
    num_kernel_runs: int,
    extra_info: Optional[dict] = None,
) -> None:
    """Print timing statistics."""
    times_us = np.array(iteration_times_ns) / 1000.0

    print(f"\nTiming results ({num_kernel_runs} runs):")
    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key}: {value}")
    print(f"  Mean: {np.mean(times_us):.2f} us")
    print(f"  Min:  {np.min(times_us):.2f} us")
    print(f"  Max:  {np.max(times_us):.2f} us")
    print(f"  Std:  {np.std(times_us):.2f} us")


def print_per_call_stats(
    iteration_times_ns: list[int],
    num_kernel_runs: int,
    calls_per_iter: int,
    num_iters: int,
) -> None:
    """Print timing statistics with per-call breakdown."""
    times_us = np.array(iteration_times_ns) / 1000.0
    total_calls = num_iters * calls_per_iter

    print(f"\nTiming results ({num_kernel_runs} runs):")
    print(f"  Mean: {np.mean(times_us):.2f} us")
    print(f"  Min:  {np.min(times_us):.2f} us")
    print(f"  Max:  {np.max(times_us):.2f} us")
    print(f"  Std:  {np.std(times_us):.2f} us")
    print(
        f"\nPer-call estimate: {np.mean(times_us) * 1000 / total_calls:.2f} ns "
        f"({total_calls} calls per kernel)"
    )


def run_nanobenchmark(
    config: NanobenchConfig,
    preprocess: Callable[[str], str],
    verify_fn: Optional[Callable] = None,
    stats_fn: Optional[Callable[[list[int]], None]] = None,
) -> Optional[list[int]]:
    """Full nanobenchmark pipeline: compile, run, print stats.

    Args:
        config: Nanobenchmark configuration
        preprocess: Function to preprocess MLIR source
        verify_fn: Optional verification function
        stats_fn: Optional custom stats function, receives iteration_times_ns

    Returns:
        Iteration times in nanoseconds, or None if GPU unavailable
    """
    hsaco_path, _ = compile_kernel(config, preprocess)
    iteration_times_ns = run_kernel(config, hsaco_path, verify_fn)

    if iteration_times_ns is not None and stats_fn is not None:
        stats_fn(iteration_times_ns)

    return iteration_times_ns
