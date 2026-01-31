"""Benchmark script for MFMA dword4 16x16x16 f16f16f32 kernel with multiple parameter configurations."""

import os
import sys
import argparse
import itertools
import multiprocessing
from typing import List, Tuple, Optional

# Add project root to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    _get_logger,
    _log_info,
)
from mlir_kernels.benchmarks.benchmark_utils import (
    BenchmarkResult,
    BaseConfig,
    format_throughput_stats,
    run_benchmark,
)
from mlir_kernels.kernel_utils import (
    MFMAConfig,
    make_mfma_preprocess,
    make_mfma_verify_fn,
    generate_mfma_data,
    LDS_SIZE_LIMIT,
)


# 304 = num CUs on MI300X
NUM_CU_PER_GPU = 304


class BenchmarkMFMAConfig(MFMAConfig, BaseConfig):
    """MFMA config that inherits from both MFMAConfig and BaseConfig for benchmarking.

    Note: MFMAConfig uses num_workgroups directly, while BaseConfig expects _num_workgroups.
    We override the property to make them compatible.
    """

    @property
    def num_workgroups(self) -> int:
        """Override to use MFMAConfig's num_workgroups field directly."""
        return self.__dict__.get("num_workgroups", self._num_workgroups)


def compile_kernel_worker(
    config: BenchmarkMFMAConfig,
) -> Tuple[BenchmarkMFMAConfig, str]:
    """Worker function for parallel compilation."""
    try:
        with ir.Context() as ctx:
            # Use shared preprocess function, but benchmark disables LDS
            def preprocess(x: str) -> str:
                x = x.replace("{{SIZE_M}}", str(config.m))
                x = x.replace("{{SIZE_N}}", str(config.n))
                x = x.replace("{{SIZE_K}}", str(config.k))
                # LDS manually disabled atm for benchmark
                x = x.replace("{{LDS_B_SHIFT}}", str(0))
                x = x.replace("{{LDS_SIZE}}", str(0))
                return x

            bench_dir = os.path.dirname(os.path.abspath(__file__))
            library_path = os.path.join(
                bench_dir, "..", "library", "common", "indexing.mlir"
            )
            asm_complete, _ = compile_mlir_file_to_asm(
                config.mlir_file,
                config.kernel_name,
                config.pass_pipeline,
                ctx,
                preprocess=preprocess,
                library_paths=[library_path],
                print_ir_after_all=False,
            )

            logger = _get_logger()
            _log_info(
                logger,
                f"[COMPILE] Assembling to HSACO: target={config.mcpu}, "
                f"wavefront_size={config.wavefront_size}",
            )
            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=config.mcpu, wavefront_size=config.wavefront_size
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")
            _log_info(
                logger,
                f"[COMPILE] HSACO assembly completed: {os.path.basename(hsaco_path)}",
            )

            return (config, hsaco_path)
    except Exception as e:
        print(
            f"\nCOMPILATION FAILED: m={config.m} n={config.n} k={config.k} "
            f"wg={config.num_workgroups} waves={config.num_waves}",
            file=sys.stderr,
        )
        raise RuntimeError(f"Failed to compile kernel with config {config}: {e}") from e


def execute_kernel_benchmark(
    config: BenchmarkMFMAConfig,
    hsaco_path: str,
    skip_test: bool = False,
    num_iterations: int = 5,
    device_id: Optional[int] = None,
) -> Tuple[Optional[BenchmarkResult], str]:
    """Execute a compiled kernel and return benchmark result with status message."""
    logger = _get_logger()

    _log_info(
        logger, f"[EXECUTE] Executing kernel: m={config.m}, n={config.n}, k={config.k}"
    )

    # Generate data using shared function
    a_data, b_data, c_data = generate_mfma_data(config)

    _log_info(
        logger,
        f"[EXECUTE] Matrices created: m={config.m}, n={config.n}, k={config.k}, batch={config.batch}",
    )

    assert all(
        sz > 0 for sz in [len(a_data), len(b_data), len(c_data)]
    ), f"All matrix sizes must be > 0 for m={config.m}, n={config.n}, k={config.k} wg={config.num_workgroups} waves={config.num_waves}"

    # Use shared verify function
    verify_fn = make_mfma_verify_fn(config) if not skip_test else None

    try:
        iteration_times_ns: List[int] = execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name=config.kernel_name,
            input_args=[a_data, b_data],
            output_args=[c_data],
            mcpu=config.mcpu,
            wavefront_size=config.wavefront_size,
            grid_dim=(config.num_workgroups, 1, 1),
            block_dim=(config.num_threads, 1, 1),
            verify_fn=verify_fn,
            num_iterations=num_iterations,
            device_id=device_id,
        )

        result: BenchmarkResult = BenchmarkResult(
            config=config, iteration_times_ns=iteration_times_ns
        )
        return result, ""
    except AssertionError as e:
        return None, f"VERIFICATION FAILED: {e}"
    except Exception as e:
        return None, f"ERROR: {e}"


def format_mfma_failure(
    config: BenchmarkMFMAConfig, error_msg: str, device_id: Optional[int]
) -> str:
    """Format failure message for MFMA benchmark."""
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    return (
        f"FAILED [{device_str}] "
        f"m={config.m:2d} n={config.n:2d} k={config.k:2d} "
        f"wg={config.num_workgroups:4d} waves={config.num_waves:2d} "
        f"lds={config.lds_total_size}: {error_msg}"
    )


def benchmark_mfma(
    configs: List[BenchmarkMFMAConfig],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[BenchmarkMFMAConfig, str]]]:
    """Benchmark multiple MFMA kernel configurations using all available GPUs.

    Jobs are distributed across GPUs in round-robin fashion with only one job running
    per GPU at a time. Control GPU visibility with CUDA_VISIBLE_DEVICES.

    Returns:
        Tuple of (successful results, failed configs with error messages)
    """
    return run_benchmark(
        configs=configs,
        compile_worker=compile_kernel_worker,
        execute_benchmark=execute_kernel_benchmark,
        num_compile_workers=num_compile_workers,
        skip_test=skip_test,
        format_failure=format_mfma_failure,
        handle_keyboard_interrupt=True,
    )


def main() -> None:
    """Main benchmark function with example configurations."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Benchmark MFMA dword4 16x16x16 f16f16f32 kernel"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip correctness verification",
    )
    args: argparse.Namespace = parser.parse_args()

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    mlir_file: str = os.path.join(
        script_dir, "..", "mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir"
    )

    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    # Problem size parameters
    m_values: List[int] = [i for i in range(3, 8)]
    n_values: List[int] = [i for i in range(3, 8)]
    k_values: List[int] = [i for i in range(3, 16)]
    # Scaling parameters
    num_workgroups_values: List[int] = [NUM_CU_PER_GPU * i for i in range(1, 10)]
    num_waves_values: List[int] = [i for i in range(1, 8)]

    # Generate all configs
    all_configs: List[BenchmarkMFMAConfig] = [
        BenchmarkMFMAConfig(
            m=m, n=n, k=k, num_workgroups=wg, num_waves=waves, mlir_file=mlir_file
        )
        for m, n, k, wg, waves in itertools.product(
            m_values, n_values, k_values, num_workgroups_values, num_waves_values
        )
    ]

    # Filter configs: limit problem size to avoid unrolling blowup and stay within LDS limit
    configs: List[BenchmarkMFMAConfig] = [
        config
        for config in all_configs
        if config.m * config.n * config.k >= 16
        and config.m * config.n * config.k <= 144
        and config.num_workgroups * config.num_waves >= 4 * NUM_CU_PER_GPU
        and config.num_workgroups * config.num_waves <= 12 * NUM_CU_PER_GPU
        and config.lds_total_size <= LDS_SIZE_LIMIT
    ]

    # Run the configurations
    results: List[BenchmarkResult]
    failed_configs: List[Tuple[BenchmarkMFMAConfig, str]]
    print(
        f"Compiling {len(configs)} configurations on {multiprocessing.cpu_count()} processes..."
    )
    try:
        results, failed_configs = benchmark_mfma(configs, skip_test=args.skip_test)
    except KeyboardInterrupt:
        print(
            "\n\nBenchmark interrupted by user during compilation. No results available.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Report the results
    if results:
        results_sorted: List[BenchmarkResult] = sorted(
            results,
            key=lambda r: (r.compute_efficiency, r.flops_per_cycle_per_wave),
            reverse=False,
        )

        print(
            "\nPerf summary (sorted by compute efficiency, lowest first):",
            file=sys.stderr,
        )
        print("=" * 140, file=sys.stderr)
        for result in results_sorted:
            config = result.config  # type: BenchmarkMFMAConfig
            print(
                f"GPU{result.device_id} "
                f"wg={config.num_workgroups:4d} waves={config.num_waves:2d} | "
                f"m={config.m:2d} n={config.n:2d} k={config.k:2d} | "
                + format_throughput_stats(result),
                file=sys.stderr,
            )
        print("=" * 140, file=sys.stderr)

    if failed_configs:
        print("\nFailed configurations:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        for config, error_msg in failed_configs:
            print(
                f"m={config.m} n={config.n} k={config.k} "
                f"wg={config.num_workgroups} waves={config.num_waves} "
                f"lds={config.lds_total_size}: {error_msg}",
                file=sys.stderr,
            )
        print("-" * 80, file=sys.stderr)

    print(f"\nSummary: {len(results)}/{len(configs)} configurations completed")
    print(f"Failures: {len(failed_configs)}")


if __name__ == "__main__":
    main()
