"""Utilities for GPU benchmarking."""

import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, TypeVar

from tqdm.auto import tqdm
from integration_test.test_utils import hip_get_device_count, DEFAULT_SROA_PASS_PIPELINE


@dataclass
class BaseConfig:
    """Base configuration class for benchmark configs.

    Provides common attributes and properties needed by BenchmarkResult. All benchmark
    config classes should inherit from this.

    Child classes must implement num_workgroups as a property or use _num_workgroups
    field.
    """

    # Required fields (must come first)
    num_waves: int
    mlir_file: str

    # Optional fields with defaults (must come after required fields)
    # Note: total_flops and total_bytes are computed properties in child classes
    _num_workgroups: int = (
        0  # Use underscore; child classes override num_workgroups property
    )
    wavefront_size: int = 64
    kernel_name: str = ""
    pass_pipeline: str = DEFAULT_SROA_PASS_PIPELINE
    mcpu: str = "gfx942"
    shader_clock_mhz: float = 2100.0
    peak_gbps: float = 5300.0
    peak_tflops: float = 1307.0

    @property
    def num_workgroups(self) -> int:
        """Number of workgroups.

        Child classes can override this property.
        """
        return self._num_workgroups

    @property
    def num_threads(self) -> int:
        """Number of threads per workgroup."""
        return self.num_waves * self.wavefront_size


# Type variable for config types
T = TypeVar("T", bound=BaseConfig)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    config: BaseConfig  # Config object (should inherit from BaseConfig) with num_threads and wavefront_size
    iteration_times_ns: List[
        int
    ]  # List of execution times in nanoseconds for each iteration
    device_id: int = 0

    @property
    def cycles(self) -> int:
        """Get the last iteration time (top-1, typically the warmest/best run)."""
        return (self.iteration_times_ns[-1] / 1e9) * (
            self.config.shader_clock_mhz * 1e6
        )

    @property
    def time_s(self) -> float:
        """Get the last iteration time in seconds."""
        return self.iteration_times_ns[-1] / 1e9

    @property
    def b_per_cycle_per_thread(self) -> float:
        """Calculate bytes per cycle per thread."""
        return self.total_bytes / self.cycles / self.config.num_threads

    @property
    def b_per_cycle_per_wave(self) -> float:
        """Calculate bytes per cycle per wave."""
        num_waves = self.config.num_threads / self.config.wavefront_size
        return self.total_bytes / self.cycles / num_waves

    @property
    def b_per_cycle_overall(self) -> float:
        """Calculate bytes per cycle overall."""
        return self.total_bytes / self.cycles

    @property
    def total_gbytes_per_second(self) -> float:
        """Total GB/s."""
        return self.total_bytes / 1e9 / self.time_s

    @property
    def total_bytes(self) -> float:
        """Total bytes."""
        return self.config.total_bytes

    @property
    def flops_per_cycle_per_wave(self) -> float:
        """Calculate FLOPS per cycle per wave."""
        num_waves = self.config.num_threads / self.config.wavefront_size
        return self.flops_per_cycle_overall / num_waves

    @property
    def flops_per_cycle_overall(self) -> float:
        """Calculate FLOPS per cycle overall."""
        return self.config.total_flops / self.cycles

    @property
    def total_flops(self) -> float:
        """Total FLOPS."""
        return self.config.total_flops

    @property
    def total_tflops_per_second(self) -> float:
        """Total TFLOPS per second."""
        return self.total_flops / 1e12 / self.time_s

    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency percentage (throughput / peak throughput)."""
        throughput_bps = self.total_bytes / self.time_s
        return throughput_bps / 1e9 / self.config.peak_gbps

    @property
    def compute_efficiency(self) -> float:
        """Calculate compute efficiency percentage (TFLOPS / peak TFLOPS)."""
        return self.total_tflops_per_second / self.config.peak_tflops


def format_throughput_stats(result: BenchmarkResult) -> str:
    """Calculate and format throughput statistics string.

    Unified function that handles both basic throughput stats and MFMA-specific
    compute metrics. Extracts all necessary information from the BenchmarkResult object.

    Args:
        result: BenchmarkResult object containing all benchmark data

    Returns:
        Formatted string with detailed performance metrics
    """
    # Optional compute throughput formatting
    flops_str = ""
    if result.config.total_flops > 0 and result.config.peak_tflops is not None:
        flops_str = (
            f"{result.flops_per_cycle_per_wave / 1e9:8.2f} GFLOP/cycle/wave | "
            f"{result.flops_per_cycle_overall / 1e9:9.2f} GFLOP/cycle overall | "
            f"{result.total_tflops_per_second:6.2f} TFLOP/s (compute_eff {result.compute_efficiency*100:5.2f}% assuming peak {result.config.peak_tflops} TFLOP/s throughput)"
        )

    # Memory throughput formatting
    nanoseconds = result.iteration_times_ns[-1]
    return (
        f"{nanoseconds:8.0f} ns ({result.cycles/1000:6.0f} kCycles), {result.total_bytes/1e3:8.0f}kB (R+W) over {result.config.num_waves:2.0f} waves | "
        f"{result.b_per_cycle_per_thread:5.2f} B/cycle/thread | "
        f"{result.b_per_cycle_per_wave:8.2f} B/cycle/wave | "
        f"{result.b_per_cycle_overall:9.2f} B/cycle overall | "
        f"{result.total_gbytes_per_second:7.2f} GB/s (mem_eff {result.memory_efficiency*100:5.2f}% assuming peak {result.config.peak_gbps} GB/s throughput) | "
        f"{flops_str}"
    )


# Global variable to store device ID for each worker process
_worker_device_id: Optional[int] = None


def _worker_init(device_id: int):
    """Initialize worker process with assigned GPU."""
    global _worker_device_id
    _worker_device_id = device_id
    # Import here to avoid issues with multiprocessing
    from integration_test.test_utils import hip_init, hip_set_device

    hip_init()
    hip_set_device(device_id)


def _run_on_device(fn_args_kwargs: tuple) -> tuple:
    """Run function on the device assigned to this worker."""
    fn, args, kwargs = fn_args_kwargs
    global _worker_device_id
    # Override device_id with worker's assigned device
    kwargs["device_id"] = _worker_device_id
    result = fn(*args, **kwargs)
    return (result, _worker_device_id)


class MultiGPUExecutor:
    """Manages execution across multiple GPUs using multiprocessing.

    Creates one worker process per GPU. Each process is initialized with
    hip_init() and hip_set_device() for its assigned GPU. Jobs are distributed
    round-robin across processes for true parallel execution.

    The submitted function must accept a `device_id` keyword argument.
    """

    def __init__(self):
        self.num_gpus = hip_get_device_count()
        if self.num_gpus < 1:
            raise RuntimeError("No GPUs available")

        # Create one ProcessPoolExecutor per GPU, each with 1 worker
        # This ensures each worker is bound to a specific GPU
        self._pools: list[ProcessPoolExecutor] = []
        for device_id in range(self.num_gpus):
            pool = ProcessPoolExecutor(
                max_workers=1,
                initializer=_worker_init,
                initargs=(device_id,),
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._pools.append(pool)

        self._next_gpu = 0

    def submit(self, fn: Callable, *args, **kwargs) -> tuple[Future, int]:
        """Submit a job to the next GPU in round-robin order.

        The function will be called with device_id=<assigned_device> added to kwargs.

        Returns:
            Tuple of (Future, device_id) - Future resolves to (result, device_id).
        """
        device_id = self._next_gpu
        self._next_gpu = (self._next_gpu + 1) % self.num_gpus

        pool = self._pools[device_id]
        future = pool.submit(_run_on_device, (fn, args, kwargs))
        return future, device_id

    def shutdown(self, wait: bool = True):
        """Shutdown all worker processes."""
        for pool in self._pools:
            pool.shutdown(wait=wait)


def run_benchmark(
    configs: List[T],
    compile_worker: Callable[[T], Tuple[T, str]],
    execute_benchmark: Callable[
        [T, str, bool, int, Optional[int]], Tuple[Optional[BenchmarkResult], str]
    ],
    num_compile_workers: int = multiprocessing.cpu_count() // 2,
    skip_test: bool = False,
    format_failure: Optional[Callable[[T, str, Optional[int]], str]] = None,
    handle_keyboard_interrupt: bool = True,
    cross_compile_only: bool = False,
) -> Tuple[List[BenchmarkResult], List[Tuple[T, str]]]:
    """Generic benchmark function for running multiple kernel configurations.

    Handles parallel compilation, multi-GPU execution, and result collection.

    Args:
        configs: List of configuration objects (must inherit from BaseConfig)
        compile_worker: Function that takes a config and returns (config, hsaco_path)
        execute_benchmark: Function that takes (config, hsaco_path, skip_test, num_iterations, device_id)
                          and returns (Optional[BenchmarkResult], str)
        num_compile_workers: Number of parallel compilation workers
        skip_test: Whether to skip correctness verification
        format_failure: Optional function to format failure messages. Takes (config, error_msg, device_id)
                        and returns a formatted string. If None, uses default formatting.
        handle_keyboard_interrupt: Whether to handle KeyboardInterrupt gracefully during execution
        cross_compile_only: Whether to avoid running (and e.g. verify everything compiles on macos)

    Returns:
        Tuple of (successful results, failed configs with error messages)
    """
    assert len(configs) > 0, "No configurations provided"

    compile_results: List[Tuple[T, str]]

    # Compile in parallel.
    try:
        with multiprocessing.Pool(processes=num_compile_workers) as pool:
            compile_results = list(
                tqdm(
                    pool.imap(compile_worker, configs),
                    total=len(configs),
                    desc="Compiling",
                    unit="kernel",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            )
    except KeyboardInterrupt:
        print("\n\nCompilation interrupted by user.", file=sys.stderr)
        raise

    if cross_compile_only:
        print("Cross-compilation only, skipping execution.")
        return [], []

    # Create multi-GPU executor.
    executor: MultiGPUExecutor = MultiGPUExecutor()
    print(
        f"\nExecuting {len(compile_results)} kernels across {executor.num_gpus} GPU(s)..."
    )
    print("=" * 80)

    # Submit all jobs
    from concurrent.futures import Future

    futures: List[Tuple[Future, T, str, int]] = []
    for i, (config, hsaco_path) in enumerate(compile_results):
        future, _ = executor.submit(execute_benchmark, config, hsaco_path, skip_test)
        futures.append((future, config, hsaco_path, i))

    # Collect results with progress bar
    results: List[BenchmarkResult] = []
    failed_configs: List[Tuple[T, str]] = []
    try:
        with tqdm(
            futures,
            desc="Benchmarking",
            unit="kernel",
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for future, config, hsaco_path, idx in pbar:
                try:
                    # Future returns ((result, message), device_id)
                    result: Optional[BenchmarkResult]
                    message: str
                    device_id: int
                    (result, message), device_id = future.result()
                    if result:
                        result.device_id = device_id
                        results.append(result)
                    else:
                        failed_configs.append((config, message))
                        if format_failure:
                            print(
                                format_failure(config, message, device_id),
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"FAILED [GPU{device_id}] {config}: {message}",
                                file=sys.stderr,
                            )
                except Exception as e:
                    error_msg: str = str(e)
                    failed_configs.append((config, error_msg))
                    if format_failure:
                        print(
                            format_failure(config, error_msg, None),
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"FAILED [GPU?] {config}: {error_msg}",
                            file=sys.stderr,
                        )
                finally:
                    # Clean up hsaco file
                    if hsaco_path and os.path.exists(hsaco_path):
                        os.unlink(hsaco_path)
                pbar.set_postfix(failures=len(failed_configs))
    except KeyboardInterrupt:
        if handle_keyboard_interrupt:
            print(
                "\n\nInterrupted by user. Cleaning up and returning partial results...",
                file=sys.stderr,
            )
            # Cancel remaining futures and clean up
            for future, config, hsaco_path, idx in futures:
                if not future.done():
                    future.cancel()
                if hsaco_path and os.path.exists(hsaco_path):
                    try:
                        os.unlink(hsaco_path)
                    except:
                        pass
            # Return partial results instead of re-raising
            executor.shutdown()
            return results, failed_configs
        else:
            raise

    executor.shutdown()
    print("=" * 80)
    return results, failed_configs
