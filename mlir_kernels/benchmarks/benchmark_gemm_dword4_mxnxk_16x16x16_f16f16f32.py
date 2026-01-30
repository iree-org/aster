"""Benchmark script for GEMM dword4 16x16x16 f16f16f32 kernel with multiple parameter configurations."""

import os
import sys
import argparse
import itertools
import multiprocessing
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

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
from aster.pass_pipelines import (
    DEFAULT_SROA_PASS_PIPELINE,
    SYNCHRONOUS_SROA_PASS_PIPELINE,
)
from mlir_kernels.benchmarks.benchmark_utils import (
    BenchmarkResult,
    BaseConfig,
    format_throughput_stats,
    run_benchmark,
)
from mlir_kernels.common import get_library_paths
from mlir_kernels.gemm_config import validate_gemm_config


# MFMA operation sizes (16x16x16)
MFMA_SIZE_M = 16
MFMA_SIZE_N = 16
MFMA_SIZE_K = 16

# LDS limit (64KB)
LDS_SIZE_LIMIT = 65536

# 304 = num CUs on MI300X
NUM_CU_PER_GPU = 304


@dataclass
class GEMMConfig(BaseConfig):
    """Configuration for GEMM 16x16x16 benchmark."""

    m: int = field(default=...)  # Problem size M
    n: int = field(default=...)  # Problem size N
    k: int = field(default=...)  # Problem size K
    m_tile: int = field(default=16)  # Tile size in M dimension
    n_tile: int = field(default=16)  # Tile size in N dimension
    k_tile: int = field(default=16)  # Tile size in K dimension
    kernel_name: str = "test_matmul_kernel"
    # BaseConfig fields: num_workgroups, num_waves, mlir_file, total_flops, total_bytes,
    # wavefront_size, pass_pipeline, mcpu, shader_clock_mhz, peak_gbps, peak_tflops

    def __post_init__(self):
        """Validate configuration using shared validation logic."""
        is_valid, error = validate_gemm_config(
            self.m,
            self.n,
            self.k,
            self.m_tile,
            self.n_tile,
            self.k_tile,
            self.num_waves,
        )
        if not is_valid:
            raise AssertionError(error)

    @property
    def num_blocks_m(self) -> int:
        """Number of blocks in M dimension."""
        return self.m // self.m_tile

    @property
    def num_blocks_n(self) -> int:
        """Number of blocks in N dimension."""
        return self.n // self.n_tile

    @property
    def num_workgroups(self) -> int:
        """Total number of workgroups (blocks)."""
        return self.num_blocks_m * self.num_blocks_n

    @property
    def num_threads(self) -> int:
        """Number of threads per workgroup."""
        return self.wavefront_size * self.num_waves

    @property
    def total_bytes(self) -> int:
        """Compute total bytes read/written."""
        size_a = np.dtype(np.float16).itemsize
        size_b = np.dtype(np.float16).itemsize
        size_c = np.dtype(np.float32).itemsize
        bytes_a = self.m * self.k * size_a
        bytes_b = self.k * self.n * size_b
        bytes_c = self.m * self.n * size_c  # WO
        return bytes_a + bytes_b + bytes_c

    @property
    def lds_a_size(self) -> int:
        """LDS size for A tile."""
        size_a = np.dtype(np.float16).itemsize
        return self.m_tile * self.k_tile * size_a

    @property
    def lds_b_size(self) -> int:
        """LDS size for B tile."""
        size_b = np.dtype(np.float16).itemsize
        return self.n_tile * self.k_tile * size_b

    @property
    def lds_total_size(self) -> int:
        """Total LDS size for A and B tiles."""
        return self.lds_a_size + self.lds_b_size

    @property
    def total_flops(self) -> int:
        """Total FLOPs for the matmul (2*M*N*K)."""
        return 2 * self.m * self.n * self.k


def compile_kernel_worker(config: GEMMConfig) -> Tuple[GEMMConfig, str]:
    """Worker function for parallel compilation."""
    try:
        with ir.Context() as ctx:
            size_a = np.dtype(np.float16).itemsize
            size_b = np.dtype(np.float16).itemsize

            def preprocess(x):
                x = x.replace("{{SIZE_M}}", str(config.m))
                x = x.replace("{{SIZE_N}}", str(config.n))
                x = x.replace("{{SIZE_K}}", str(config.k))
                x = x.replace("{{TILE_SIZE_M}}", str(config.m_tile))
                x = x.replace("{{TILE_SIZE_N}}", str(config.n_tile))
                x = x.replace("{{TILE_SIZE_K}}", str(config.k_tile))
                x = x.replace("{{NUM_BLOCKS}}", str(config.num_workgroups))
                x = x.replace("{{NUM_THREADS}}", str(config.num_threads))
                x = x.replace("{{LDS_SIZE}}", str(config.lds_total_size))
                SIZE_K_BY_TILE_SIZE_K = config.k // config.k_tile
                x = x.replace("{{SIZE_K_BY_TILE_SIZE_K}}", str(SIZE_K_BY_TILE_SIZE_K))
                # Perform replacement for LOOP_SIZE_D_MMNNKK (proper count needed for
                # scheduling to kick in).
                mnkt = config.m_tile * config.n_tile * config.k_tile
                mnk_mfma = MFMA_SIZE_M * MFMA_SIZE_N * MFMA_SIZE_K
                mnkt_mfma = mnkt // mnk_mfma
                LOOP_SIZE_D_MMNNKK = mnkt_mfma // config.num_waves
                # These should have been checked by validate_gemm_config; this is a
                # sanity check.
                assert mnkt % mnk_mfma == 0, "Invalid configuration"
                assert mnkt_mfma % config.num_waves == 0, "Invalid configuration"
                x = x.replace("{{LOOP_SIZE_D_MMNNKK}}", str(LOOP_SIZE_D_MMNNKK))
                return x

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
    config: GEMMConfig,
    hsaco_path: str,
    skip_test: bool = False,
    num_iterations: int = 5,
    device_id: Optional[int] = None,
) -> Tuple[Optional[BenchmarkResult], str]:
    """Execute a compiled kernel and return benchmark result with status message."""
    logger = _get_logger()
    dt_a: type = np.float16
    dt_b: type = np.float16
    dt_c: type = np.float32

    _log_info(
        logger, f"[EXECUTE] Executing kernel: m={config.m}, n={config.n}, k={config.k}"
    )

    # Create matrices in standard row-major layout
    # Values in [-1.0, -0.5] ∪ [0.5, 1.0] for well-conditioned matrices
    mean = 0.75
    a_data: np.ndarray = (
        np.random.uniform(mean * 2 / 3, mean * 4 / 3, (config.m, config.k))
        * np.random.choice([-1, 1], (config.m, config.k))
    ).astype(dt_a)
    b_data: np.ndarray = (
        np.random.uniform(mean * 2 / 3, mean * 4 / 3, (config.k, config.n))
        * np.random.choice([-1, 1], (config.k, config.n))
    ).astype(dt_b)
    c_data: np.ndarray = np.zeros(config.m * config.n, dtype=dt_c)

    _log_info(
        logger,
        f"[EXECUTE] Matrices created: m={config.m}, n={config.n}, k={config.k}",
    )

    def verify_fn(input_args: List[np.ndarray], output_args: List[np.ndarray]) -> None:
        a_flat = np.array(input_args[0])
        a = a_flat.reshape(config.m, config.k)
        b_flat = np.array(input_args[1])
        # B is transposed in the kernel (stored as N x K)
        b = b_flat.reshape(config.n, config.k).T
        c_flat = np.array(output_args[0], dtype=dt_c)
        c = c_flat.reshape(config.m, config.n)
        ref_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))

        # Scale tolerance with problem size
        # f16 has 10-bit mantissa → ~1e-3 relative precision
        # Error accumulates with k reductions (sqrt(k) factor)
        rtol = 1e-3 * np.sqrt(config.k)
        atol = 1e-3 * config.k * mean
        if not np.allclose(c, ref_f32, rtol=rtol, atol=atol):
            diff = np.abs(c.astype(np.float32) - ref_f32)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            max_diff = diff[max_idx]
            relative_error = np.linalg.norm(diff) / np.linalg.norm(ref_f32)
            raise AssertionError(
                f"GEMM kernel failed!\n"
                f"Max diff: {max_diff} at {max_idx}, c={c[max_idx]}, ref={ref_f32[max_idx]}\n"
                f"Relative error: {relative_error}, rtol: {rtol}, atol: {atol}\n"
                f"c shape: {c.shape}, ref shape: {ref_f32.shape}"
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
            verify_fn=verify_fn if not skip_test else None,
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
    config: GEMMConfig, error_msg: str, device_id: Optional[int]
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
    configs: List[GEMMConfig],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[GEMMConfig, str]]]:
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
        default="gemm_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        # default="gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        help="MLIR filename (default: gemm_dword4_mxnxk_16x16x16_f16f16f32.mlir)",
    )
    args: argparse.Namespace = parser.parse_args()

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    mlir_file: str = os.path.join(script_dir, "..", args.mlir_filename)

    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    # Problem size parameters (powers of 2 for typical GEMM sizes)
    # These are actual matrix dimensions, not block counts
    m_values: List[int] = [128, 256, 512, 1024, 2048, 2048 * 8]
    n_values: List[int] = [128, 256, 512, 1024, 2048, 2048 * 8]
    k_values: List[int] = [128, 256, 512, 1024]

    # Tile sizes (must divide problem dimensions evenly, and be multiples of 16)
    tile_configs: List[Tuple[int, int, int]] = [
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
    num_waves_values: List[int] = args.num_waves

    # Generate all valid configs
    all_configs: List[GEMMConfig] = []
    for m, n, k in itertools.product(m_values, n_values, k_values):
        for m_tile, n_tile, k_tile in tile_configs:
            # Skip invalid tile configurations
            if m % m_tile != 0 or n % n_tile != 0 or k % k_tile != 0:
                continue
            for num_waves in num_waves_values:
                try:
                    config = GEMMConfig(
                        m=m,
                        n=n,
                        k=k,
                        m_tile=m_tile,
                        n_tile=n_tile,
                        k_tile=k_tile,
                        num_waves=num_waves,
                        mlir_file=mlir_file,
                        # pass_pipeline=SYNCHRONOUS_SROA_PASS_PIPELINE,
                        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
                    )
                    all_configs.append(config)
                except AssertionError:
                    continue

    # Filter configs: stay within LDS limit and reasonable problem sizes
    configs: List[GEMMConfig] = [
        config
        for config in all_configs
        if config.lds_total_size <= LDS_SIZE_LIMIT
        and config.total_flops >= 2 * 128 * 128 * 128  # Minimum problem size
        and config.total_flops <= 2 * 2048 * 2048 * 2048  # Maximum problem size
        # Maximum FLOPS per wave (in MFMA operations) to avoid too much unrolling atm
        and config.total_flops / (config.num_waves * config.num_workgroups)
        <= 1024 * (MFMA_SIZE_M * MFMA_SIZE_N * MFMA_SIZE_K)
    ]

    print(f"Generated {len(configs)} valid configurations")

    if len(configs) == 0:
        print("No valid configurations found!")
        return

    # Run the configurations
    results: List[BenchmarkResult]
    failed_configs: List[Tuple[GEMMConfig, str]]
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
            config: GEMMConfig = result.config
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
