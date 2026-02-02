"""Common utilities for integration tests.

This module re-exports from aster.testing for backwards compatibility. New code should
import directly from aster.testing.
"""

# Re-export everything from aster.testing
from aster.testing import (
    # Constants
    DEFAULT_MCPU,
    DEFAULT_WAVEFRONT_SIZE,
    # Logging
    MillisecondFormatter,
    _get_logger,
    _should_log,
    _log_info,
    _log_with_device,
    # Context managers
    hsaco_file,
    # Compilation
    load_mlir_module_from_file,
    compile_mlir_file_to_asm,
    # Execution
    execute_kernel_and_verify,
    compile_and_run,
)

# Re-export HIP runtime functions that benchmarks use
from aster._mlir_libs._runtime_module import (
    hip_init,
    hip_get_device_count,
    hip_set_device,
    hip_get_device,
    hip_device_synchronize,
    hip_malloc,
    hip_free,
    hip_memcpy_host_to_device,
    hip_memcpy_device_to_host,
)

# Keep FlushLLC in integration_test since it's specific to benchmarking
from integration_test.flush_llc import FlushLLC

__all__ = [
    # Constants
    "DEFAULT_MCPU",
    "DEFAULT_WAVEFRONT_SIZE",
    # Logging
    "MillisecondFormatter",
    "_get_logger",
    "_should_log",
    "_log_info",
    "_log_with_device",
    # Context managers
    "hsaco_file",
    # Compilation
    "load_mlir_module_from_file",
    "compile_mlir_file_to_asm",
    # Execution
    "execute_kernel_and_verify",
    "compile_and_run",
    # HIP runtime
    "hip_init",
    "hip_get_device_count",
    "hip_set_device",
    "hip_get_device",
    "hip_device_synchronize",
    "hip_malloc",
    "hip_free",
    "hip_memcpy_host_to_device",
    "hip_memcpy_device_to_host",
    # Benchmarking
    "FlushLLC",
]
