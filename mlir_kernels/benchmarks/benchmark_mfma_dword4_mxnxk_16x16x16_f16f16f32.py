"""Benchmark script for MFMA dword4 16x16x16 f16f16f32 kernel with multiple parameter configurations."""

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
from mlir_kernels.benchmarks.benchmark_utils import (
    BenchmarkResult,
    BaseConfig,
    format_throughput_stats,
    run_benchmark,
)


# Block sizes for each MFMA operation dimension (16x16x16)
M_BLOCK_SIZE = 16
N_BLOCK_SIZE = 16
K_BLOCK_SIZE = 16

# LDS limit (64KB)
LDS_SIZE_LIMIT = 65536

# 304 = num CUs on MI300X
NUM_CU_PER_GPU = 304


@dataclass
class MFMAConfig(BaseConfig):
    """Configuration for MFMA 16x16x16 benchmark."""

    m: int = field(default=...)  # Number of blocks in M dimension
    n: int = field(default=...)  # Number of blocks in N dimension
    k: int = field(default=...)  # Number of blocks in K dimension
    kernel_name: str = "test_matmul_kernel"
    # BaseConfig fields: num_workgroups, num_waves, mlir_file, total_flops, total_bytes,
    # wavefront_size, pass_pipeline, mcpu, shader_clock_mhz, peak_gbps, peak_tflops

    @property
    def total_bytes(self) -> int:
        """Compute total bytes read/written."""
        size_a = np.dtype(np.float16).itemsize
        size_b = np.dtype(np.float16).itemsize
        size_c = np.dtype(np.float32).itemsize
        bytes_a = self.m * self.k * M_BLOCK_SIZE * K_BLOCK_SIZE * size_a
        bytes_b = self.k * self.n * K_BLOCK_SIZE * N_BLOCK_SIZE * size_b
        bytes_c = self.m * self.n * M_BLOCK_SIZE * N_BLOCK_SIZE * size_c  # WO
        return (bytes_a + bytes_b + bytes_c) * self.num_waves * self.num_workgroups

    @property
    def lds_a_size(self) -> int:
        """LDS size for A matrix (all waves colocate in shared memory)."""
        size_a = np.dtype(np.float16).itemsize
        return self.m * self.k * M_BLOCK_SIZE * K_BLOCK_SIZE * size_a * self.num_waves

    @property
    def lds_b_size(self) -> int:
        """LDS size for B matrix (all waves colocate in shared memory)."""
        size_b = np.dtype(np.float16).itemsize
        return self.k * self.n * K_BLOCK_SIZE * N_BLOCK_SIZE * size_b * self.num_waves

    @property
    def lds_total_size(self) -> int:
        """Total LDS size for A and B matrices."""
        return self.lds_a_size + self.lds_b_size

    @property
    def total_flops(self) -> int:
        """Total FLOPs for the matmul."""
        flops_per_wave = (
            self.m * self.n * self.k * 2 * M_BLOCK_SIZE * N_BLOCK_SIZE * K_BLOCK_SIZE
        )
        return flops_per_wave * self.num_waves * self.num_workgroups


def compile_kernel_worker(config: MFMAConfig) -> Tuple[MFMAConfig, str]:
    """Worker function for parallel compilation."""
    try:
        with ir.Context() as ctx:

            def preprocess(x: str) -> str:
                x = x.replace("{{SIZE_M}}", str(config.m))
                x = x.replace("{{SIZE_N}}", str(config.n))
                x = x.replace("{{SIZE_K}}", str(config.k))
                # x = x.replace("{{LDS_B_SHIFT}}", str(config.lds_a_size))
                # x = x.replace("{{LDS_SIZE}}", str(config.lds_total_size))
                # LDS manually disabled atm
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
    config: MFMAConfig,
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
    # Create matrices with block-major layout
    # batch by num_workgroups * num_waves since each workgroup/wave needs its own data
    batch: int = config.num_workgroups * config.num_waves
    a_size: int = batch * (config.m * config.k) * (M_BLOCK_SIZE * K_BLOCK_SIZE)
    b_size: int = batch * (config.k * config.n) * (K_BLOCK_SIZE * N_BLOCK_SIZE)
    c_size: int = batch * (config.m * config.n) * (M_BLOCK_SIZE * N_BLOCK_SIZE)
    a_data: np.ndarray = np.full(a_size, 1.0, dtype=dt_a)
    b_data: np.ndarray = np.full(b_size, 2.0, dtype=dt_b)
    c_data: np.ndarray = np.zeros(c_size, dtype=dt_c)
    _log_info(
        logger,
        f"[EXECUTE] Matrices created: m={config.m}, n={config.n}, k={config.k}, batch={batch}",
    )

    assert all(
        sz > 0 for sz in [a_size, b_size, c_size]
    ), f"All matrix sizes must be > 0 for m={config.m}, n={config.n}, k={config.k} wg={config.num_workgroups} waves={config.num_waves}"

    def verify_fn(input_args: List[np.ndarray], output_args: List[np.ndarray]) -> None:
        # Convert from block-major to element-major layout for verification
        a_flat = np.array(input_args[0])
        a_blocks = a_flat.reshape(batch, config.m, config.k, M_BLOCK_SIZE, K_BLOCK_SIZE)

        b_flat = np.array(input_args[1])
        b_blocks = b_flat.reshape(batch, config.k, config.n, K_BLOCK_SIZE, N_BLOCK_SIZE)

        c_flat = np.array(output_args[0])
        c_blocks = c_flat.reshape(batch, config.m, config.n, M_BLOCK_SIZE, N_BLOCK_SIZE)

        # Compute reference using block matrix multiplication
        ref = np.zeros(
            (batch, config.m, config.n, M_BLOCK_SIZE, N_BLOCK_SIZE), dtype=dt_c
        )
        for b in range(batch):
            for i in range(config.m):
                for j in range(config.n):
                    for l in range(config.k):
                        a_block = a_blocks[b, i, l]
                        b_block = b_blocks[b, l, j]
                        ref[b, i, j] = ref[b, i, j] + np.matmul(
                            a_block.astype(dt_c), b_block.astype(dt_c)
                        )

        if not np.allclose(c_blocks, ref, rtol=1e-5, atol=1e-5):
            diff = np.abs(c_blocks - ref)
            max_diff = np.max(diff)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"MFMA kernel failed! Max diff: {max_diff} at index {max_idx}\n"
                f"c shape: {c_blocks.shape}, ref shape: {ref.shape}\n"
                f"c_blocks:\n{c_blocks}\nref:\n{ref}"
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


def format_mfma_failure(
    config: MFMAConfig, error_msg: str, device_id: Optional[int]
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
    configs: List[MFMAConfig],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[MFMAConfig, str]]]:
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
    all_configs: List[MFMAConfig] = [
        MFMAConfig(
            m=m, n=n, k=k, _num_workgroups=wg, num_waves=waves, mlir_file=mlir_file
        )
        for m, n, k, wg, waves in itertools.product(
            m_values, n_values, k_values, num_workgroups_values, num_waves_values
        )
    ]

    # Filter configs: limit problem size to avoid unrolling blowup and stay within LDS limit
    configs: List[MFMAConfig] = [
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
    failed_configs: List[Tuple[MFMAConfig, str]]
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
            config: MFMAConfig = result.config
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
