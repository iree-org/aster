"""Benchmark script for GEMM dword4 16x16x16 f16f16f32 kernel with multiple parameter configurations."""

import os
import sys
import argparse
import itertools
import multiprocessing
from typing import List, Tuple, Optional

import numpy as np
from aster import ir, utils
from aster.testing import (
    compile_mlir_file_to_asm,
    _get_logger,
    _log_info,
)
from aster.pass_pipelines import (
    DEFAULT_SROA_PASS_PIPELINE,
    TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
)
from mlir_kernels.benchmarks.benchmark_utils import (
    BenchmarkResult,
    BaseConfig,
    format_throughput_stats,
    run_benchmark,
)
from mlir_kernels.common import get_library_paths
from mlir_kernels.kernel_utils import (
    GEMMConfig,
    MFMA_SIZE,
    make_gemm_preprocess,
    make_gemm_verify_fn,
    generate_gemm_data,
    LDS_SIZE_LIMIT,
)
from aster.testing import execute_kernel_and_verify

# 304 = num CUs on MI300X
NUM_CU_PER_GPU = 304


class BenchmarkGEMMConfig(GEMMConfig, BaseConfig):
    """GEMM config that inherits from both GEMMConfig and BaseConfig for benchmarking."""

    def __post_init__(self):
        # Call GEMMConfig's validation
        GEMMConfig.__post_init__(self)


def compile_kernel_worker(
    config: BenchmarkGEMMConfig,
) -> Tuple[BenchmarkGEMMConfig, str]:
    """Worker function for parallel compilation."""
    try:
        with ir.Context() as ctx:
            preprocess = make_gemm_preprocess(config)
            library_paths = get_library_paths()

            asm_complete, _ = compile_mlir_file_to_asm(
                config.mlir_file,
                config.kernel_name,
                config.pass_pipeline,
                ctx,
                preprocess=preprocess,
                library_paths=library_paths,
                print_ir_after_all=False,
                print_timings=False,
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
            f"tiles=({config.m_tile},{config.n_tile},{config.k_tile}) "
            f"waves={config.num_waves}",
            file=sys.stderr,
        )
        raise RuntimeError(f"Failed to compile kernel with config {config}: {e}") from e


def execute_kernel_benchmark(
    config: BenchmarkGEMMConfig,
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

    # Generate data with well-conditioned values for benchmarks
    a_data, b_data, c_data = generate_gemm_data(config, random_data=False)

    _log_info(
        logger,
        f"[EXECUTE] Matrices created: m={config.m}, n={config.n}, k={config.k}",
    )

    # Use scaled tolerance for large k
    verify_fn = (
        make_gemm_verify_fn(config, scale_with_k=True) if not skip_test else None
    )

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


def format_gemm_failure(
    config: BenchmarkGEMMConfig, error_msg: str, device_id: Optional[int]
) -> str:
    """Format failure message for GEMM benchmark."""
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    return (
        f"FAILED [{device_str}] "
        f"m={config.m:4d} n={config.n:4d} k={config.k:4d} "
        f"tiles=({config.m_tile:2d},{config.n_tile:2d},{config.k_tile:2d}) "
        f"waves={config.num_waves:2d} "
        f"lds={config.lds_total_size}: {error_msg}"
    )


def benchmark_gemm(
    configs: List[BenchmarkGEMMConfig],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[BenchmarkGEMMConfig, str]]]:
    """Benchmark multiple GEMM kernel configurations using all available GPUs.

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
        format_failure=format_gemm_failure,
        handle_keyboard_interrupt=True,
        cross_compile_only=False,
    )


def main() -> None:
    """Main benchmark function with example configurations."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Benchmark GEMM dword4 16x16x16 f16f16f32 kernel"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Number of waves per block (default: [1, 2, 4])",
    )
    parser.add_argument(
        "--mlir-filename",
        type=str,
        default="gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        help="MLIR filename (default: gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal configuration for quick validation",
    )
    args: argparse.Namespace = parser.parse_args()

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    mlir_file: str = os.path.join(script_dir, "..", args.mlir_filename)

    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    # Problem size parameters (powers of 2 for typical GEMM sizes)
    if args.smoke_test:
        # Minimal config for smoke test
        m_values: List[int] = [128]
        n_values: List[int] = [128]
        k_values: List[int] = [128]
        tile_configs: List[Tuple[int, int, int]] = [(16, 16, 16)]
        num_waves_values: List[int] = [1]
    else:
        # These are actual matrix dimensions, not block counts
        m_values = [128, 256, 512, 1024, 2048, 2048 * 8]
        n_values = [128, 256, 512, 1024, 2048, 2048 * 8]
        k_values = [128, 256, 512, 1024]
        # Tile sizes (must divide problem dimensions evenly, and be multiples of 16)
        tile_configs = [
            (16, 16, 16),
            (32, 16, 16),
            (16, 32, 16),
            (32, 32, 16),
            (32, 32, 32),
            (32, 64, 64),
            (64, 32, 64),
            (64, 64, 32),
            (64, 64, 64),
        ]
        # Number of waves per block
        num_waves_values = args.num_waves

    # Generate all valid configs
    all_configs: List[BenchmarkGEMMConfig] = []
    for m, n, k in itertools.product(m_values, n_values, k_values):
        for m_tile, n_tile, k_tile in tile_configs:
            # Skip invalid tile configurations
            if m % m_tile != 0 or n % n_tile != 0 or k % k_tile != 0:
                continue
            for num_waves in num_waves_values:
                try:
                    config = BenchmarkGEMMConfig(
                        m=m,
                        n=n,
                        k=k,
                        m_tile=m_tile,
                        n_tile=n_tile,
                        k_tile=k_tile,
                        num_waves=num_waves,
                        mlir_file=mlir_file,
                        # pass_pipeline=TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
                        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
                    )
                    all_configs.append(config)
                except ValueError:
                    continue

    # Filter configs: stay within LDS limit and reasonable problem sizes
    configs: List[BenchmarkGEMMConfig] = [
        config
        for config in all_configs
        if config.lds_total_size <= LDS_SIZE_LIMIT
        and config.total_flops >= 2 * 128 * 128 * 128  # Minimum problem size
        and config.total_flops <= 2 * 2048 * 2048 * 2048  # Maximum problem size
        # Maximum FLOPS per wave (in MFMA operations) to avoid too much unrolling atm
        and config.total_flops / (config.num_waves * config.num_workgroups)
        <= 1024 * MFMA_SIZE
    ]

    print(f"Generated {len(configs)} valid configurations")

    if len(configs) == 0:
        print("No valid configurations found!")
        return

    # Run the configurations
    results: List[BenchmarkResult]
    failed_configs: List[Tuple[BenchmarkGEMMConfig, str]]
    print(
        f"Compiling {len(configs)} configurations on {multiprocessing.cpu_count()} processes..."
    )
    try:
        results, failed_configs = benchmark_gemm(configs, skip_test=args.skip_test)
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
        print("=" * 160, file=sys.stderr)
        for result in results_sorted:
            config = result.config  # type: BenchmarkGEMMConfig
            print(
                f"GPU{result.device_id} "
                f"m={config.m:4d} n={config.n:4d} k={config.k:4d} | "
                f"tiles=({config.m_tile:2d},{config.n_tile:2d},{config.k_tile:2d}) "
                f"workgroups={config.num_workgroups:2d} | "
                f"waves={config.num_waves:2d} | " + format_throughput_stats(result),
                file=sys.stderr,
            )
        print("=" * 160, file=sys.stderr)

    if failed_configs:
        print("\nFailed configurations:", file=sys.stderr)
        print("-" * 100, file=sys.stderr)
        for config, error_msg in failed_configs:
            print(
                f"m={config.m} n={config.n} k={config.k} "
                f"tiles=({config.m_tile},{config.n_tile},{config.k_tile}) "
                f"waves={config.num_waves} "
                f"lds={config.lds_total_size}: {error_msg}",
                file=sys.stderr,
            )
        print("-" * 100, file=sys.stderr)

    print(f"\nSummary: {len(results)}/{len(configs)} configurations completed")
    print(f"Failures: {len(failed_configs)}")


if __name__ == "__main__":
    main()
