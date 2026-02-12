"""Benchmark script for copy_1d_dwordx4 kernel with multiple parameter configurations."""

import os
import sys
import argparse
import itertools
import multiprocessing
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
from aster import ir, utils
from aster.testing import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
)
from aster.testing.flush_llc import FlushLLC
from mlir_kernels.benchmarks.benchmark_utils import (
    BenchmarkResult,
    BaseConfig,
    format_throughput_stats,
    run_benchmark,
)


@dataclass
class Copy1DConfig(BaseConfig):
    """Configuration for copy_1d_dwordx4 benchmark."""

    num_elements_per_thread: int = field(default=...)
    element_size: int = 16
    sched_delay_store: int = 3
    padding_bytes: Optional[List[int]] = None
    kernel_name: str = "copy_1d_dwordx4_static"
    # BaseConfig fields: num_workgroups, num_waves, mlir_file, total_flops, total_bytes,
    # wavefront_size, pass_pipeline, mcpu, shader_clock_mhz, peak_gbps, peak_tflops

    @property
    def total_num_elements_as_int32(self) -> int:
        """Total number of int32 elements needed for input/output arrays."""
        assert (
            self.element_size % np.dtype(np.int32).itemsize == 0
        ), "element_size must be divisible by int32 itemsize"
        return (
            self.num_workgroups
            * self.num_elements_per_thread
            * self.num_threads
            * self.element_size
        ) // np.dtype(np.int32).itemsize

    @property
    def total_bytes(self) -> int:
        """Total bytes read + written."""
        return 2 * self.total_num_elements_as_int32 * np.dtype(np.int32).itemsize

    @property
    def total_flops(self) -> int:
        """Total FLOPs (zero for copy operation)."""
        return 0


def compile_kernel_worker(config: Copy1DConfig) -> Tuple[Copy1DConfig, str]:
    """Worker function for parallel compilation."""
    try:
        with ir.Context() as ctx:

            def preprocess(x: str) -> str:
                x = x.replace(
                    "{{NUM_ELEMENTS_PER_THREAD}}", str(config.num_elements_per_thread)
                )
                x = x.replace(
                    "{{BLOCK_DIM_X}}", str(config.num_waves * config.wavefront_size)
                )
                x = x.replace("{{SCHED_DELAY_STORE}}", str(config.sched_delay_store))
                x = x.replace("{{GRID_DIM_X}}", str(config.num_workgroups))
                return x

            # Get library paths relative to the MLIR file (use absolute paths)
            # Load dependencies first: register-init and indexing before copies
            mlir_dir = os.path.dirname(os.path.abspath(config.mlir_file))
            library_dir = os.path.join(mlir_dir, "library", "common")
            library_paths = [
                os.path.abspath(os.path.join(library_dir, "register-init.mlir")),
                os.path.abspath(os.path.join(library_dir, "indexing.mlir")),
                os.path.abspath(os.path.join(library_dir, "copies.mlir")),
            ]

            asm_complete, _ = compile_mlir_file_to_asm(
                config.mlir_file,
                config.kernel_name,
                config.pass_pipeline,
                ctx,
                preprocess=preprocess,
                library_paths=library_paths,
            )

            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=config.mcpu, wavefront_size=config.wavefront_size
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")

            return (config, hsaco_path)
    except Exception as e:
        raise RuntimeError(f"Failed to compile kernel with config {config}: {e}") from e


def execute_kernel_benchmark(
    config: Copy1DConfig,
    hsaco_path: str,
    skip_test: bool = False,
    num_iterations: int = 5,
    device_id: Optional[int] = None,
) -> Tuple[Optional[BenchmarkResult], str]:
    """Execute a compiled kernel and return benchmark result with status message."""
    input_data: np.ndarray = np.arange(
        config.total_num_elements_as_int32, dtype=np.int32
    )
    output_data: np.ndarray = np.zeros(
        config.total_num_elements_as_int32, dtype=np.int32
    )
    timing_buffer_begin: np.ndarray = np.zeros(1, dtype=np.int64)
    timing_buffer_end: np.ndarray = np.zeros(1, dtype=np.int64)

    def verify_fn(input_args: List[np.ndarray], output_args: List[np.ndarray]) -> None:
        expected = input_args[0]
        actual = output_args[0]
        if not np.array_equal(expected, actual):
            diff_indices = np.where(expected != actual)[0]
            first_diff_idx = diff_indices[0] if len(diff_indices) > 0 else None
            if first_diff_idx is not None:
                raise AssertionError(
                    f"Copy kernel failed at index {first_diff_idx}: "
                    f"expected {expected[first_diff_idx]}, got {actual[first_diff_idx]}"
                )
            else:
                raise AssertionError("Copy kernel failed!")

    try:
        iteration_times_ns: List[int] = execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name=config.kernel_name,
            input_args=[input_data],
            output_args=[output_data, timing_buffer_begin, timing_buffer_end],
            mcpu=config.mcpu,
            wavefront_size=config.wavefront_size,
            grid_dim=(config.num_workgroups, 1, 1),
            block_dim=(config.num_threads, 1, 1),
            verify_fn=verify_fn if not skip_test else None,
            num_iterations=num_iterations,
            device_id=device_id,
            # Flush LLC between iterations
            flush_llc=FlushLLC(mcpu=config.mcpu),
        )

        result: BenchmarkResult = BenchmarkResult(
            config=config, iteration_times_ns=iteration_times_ns
        )
        return result, ""
    except AssertionError as e:
        return None, f"VERIFICATION FAILED: {e}"
    except Exception as e:
        return None, f"ERROR: {e}"


def format_copy_failure(
    config: Copy1DConfig, error_msg: str, device_id: Optional[int]
) -> str:
    """Format failure message for copy benchmark."""
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    padding_str = str(config.padding_bytes) if config.padding_bytes else "[0, 0]"
    return (
        f"FAILED [{device_str}] "
        f"wg={config.num_workgroups:5d} waves={config.num_waves:3d} "
        f"elems={config.num_elements_per_thread:4d} "
        f"delay={config.sched_delay_store} "
        f"padding={padding_str}: {error_msg}"
    )


def benchmark_copy_1d(
    configs: List[Copy1DConfig],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[Copy1DConfig, str]]]:
    """Benchmark multiple copy_1d kernel configurations using all available GPUs.

    Jobs are distributed across GPUs in round-robin fashion with only one job running
    per GPU at a time. Control GPU visibility with CUDA_VISIBLE_DEVICES.
    """
    return run_benchmark(
        configs=configs,
        compile_worker=compile_kernel_worker,
        execute_benchmark=execute_kernel_benchmark,
        num_compile_workers=num_compile_workers,
        skip_test=skip_test,
        format_failure=format_copy_failure,
        handle_keyboard_interrupt=False,
    )


def main() -> None:
    """Main benchmark function with example configurations."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Benchmark copy_1d_dwordx4 kernel"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal configuration for quick validation",
    )
    args: argparse.Namespace = parser.parse_args()

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    mlir_file: str = os.path.join(script_dir, "..", "copy-1d-dwordx4.mlir")

    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    # Define benchmark configurations
    if args.smoke_test:
        # Minimal config for smoke test
        num_workgroups_values: List[int] = [1]
        num_waves_values: List[int] = [1]
        num_elements_per_thread_values: List[int] = [4]
        sched_delay_store_values: List[int] = [0]
        padding_bytes_values: List[List[int]] = [[0, 0]]
    else:
        num_workgroups_values = [1, 304, 608, 3040]
        num_waves_values = [1, 2, 5, 8, 10, 16]
        num_elements_per_thread_values = [1, 4, 6, 8, 12, 16]
        sched_delay_store_values = [0, 3, 8]
        padding_bytes_values = [[0, 0], [1, 1], [2, 2], [3, 3]]
    configs: List[Copy1DConfig] = [
        Copy1DConfig(
            _num_workgroups=num_workgroups,
            num_waves=num_waves,
            num_elements_per_thread=num_elems,
            sched_delay_store=sched_delay,
            padding_bytes=padding,
            mlir_file=mlir_file,
        )
        for num_workgroups, num_waves, num_elems, sched_delay, padding in itertools.product(
            num_workgroups_values,
            num_waves_values,
            num_elements_per_thread_values,
            sched_delay_store_values,
            padding_bytes_values,
        )
    ]

    # Run the configurations
    results: List[BenchmarkResult]
    failed_configs: List[Tuple[Copy1DConfig, str]]
    results, failed_configs = benchmark_copy_1d(configs, skip_test=args.skip_test)

    # Report the results
    if results:
        results_sorted: List[BenchmarkResult] = sorted(
            results,
            key=lambda r: (r.memory_efficiency, r.b_per_cycle_per_wave),
            reverse=False,
        )

        print(
            "\nPerf summary (sorted by global efficiency, lowest first):",
            file=sys.stderr,
        )
        print("=" * 80, file=sys.stderr)
        for result in results_sorted:
            config: Copy1DConfig = result.config
            padding_str: str = (
                str(config.padding_bytes) if config.padding_bytes else "[0, 0]"
            )
            print(
                f"GPU{result.device_id} "
                f"wg={config.num_workgroups:5d}, "
                f"waves={config.num_waves:3d}, "
                f"elems={config.num_elements_per_thread:4d}, "
                f"delay={config.sched_delay_store}, "
                f"padding={padding_str}: " + format_throughput_stats(result),
                file=sys.stderr,
            )
        print("=" * 80, file=sys.stderr)

    if failed_configs:
        print("\nFailed configurations:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        for config, error_msg in failed_configs:
            padding_str: str = (
                str(config.padding_bytes) if config.padding_bytes else "[0, 0]"
            )
            print(
                f"wg={config.num_workgroups:5d} waves={config.num_waves:3d} "
                f"elems={config.num_elements_per_thread:4d} "
                f"delay={config.sched_delay_store} "
                f"padding={padding_str}: {error_msg}",
                file=sys.stderr,
            )
        print("-" * 80, file=sys.stderr)

    print(f"\nSummary: {len(results)}/{len(configs)} configurations completed")
    print(f"Failures: {len(failed_configs)}")


if __name__ == "__main__":
    main()
