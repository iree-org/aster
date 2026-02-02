"""Kernel execution utilities for aster."""

import os
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Callable, Optional, List, Any, Generator

from aster import utils
from aster.logging import _get_logger, _log_with_device
from aster._mlir_libs._runtime_module import (
    hip_init,
    hip_module_load_data,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_device_synchronize,
    hip_free,
    hip_malloc,
    hip_memcpy_host_to_device,
    hip_memcpy_device_to_host,
    hip_module_unload,
    hip_function_free,
    hip_get_device_count,
    hip_set_device,
    hip_get_device,
    hip_event_create,
    hip_event_destroy,
    hip_event_record,
    hip_event_synchronize,
    hip_event_elapsed_time,
)

# Default execution configuration
DEFAULT_MCPU = "gfx942"
DEFAULT_WAVEFRONT_SIZE = 64

__all__ = [
    "DEFAULT_MCPU",
    "DEFAULT_WAVEFRONT_SIZE",
    "hsaco_file",
    "execute_kernel_and_verify",
]


@contextmanager
def hsaco_file(path: str) -> Generator[str, None, None]:
    """Context manager that cleans up an HSACO file on exit.

    Args:
        path: Path to the HSACO file

    Yields:
        The path to the HSACO file
    """
    try:
        yield path
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def execute_kernel_and_verify(
    hsaco_path: Optional[str],
    kernel_name: str,
    input_args: List[np.ndarray],
    output_args: List[np.ndarray],
    mcpu: str,
    wavefront_size: int = 64,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    verify_fn: Optional[Callable[[List[np.ndarray], List[np.ndarray]], None]] = None,
    padding_bytes: Optional[List[int]] = None,
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
) -> List[int]:
    """Execute a GPU kernel and verify its results.

    Args:
        hsaco_path: Path to the HSACO file
        kernel_name: Name of the kernel function
        input_args: List of input numpy arrays
        output_args: List of output numpy arrays (will be modified in-place)
        mcpu: Target GPU architecture (e.g., "gfx942")
        wavefront_size: Wavefront size (default: 64)
        grid_dim: Grid dimensions (default: (1, 1, 1))
        block_dim: Block dimensions (default: (64, 1, 1))
        verify_fn: Custom verification function (default: None)
        padding_bytes: List of padding bytes per buffer (default: None)
        num_iterations: Number of times to execute the kernel (default: 1)
        device_id: GPU device ID to use (default: None, uses current device)
        flush_llc: Optional FlushLLC instance for cache flushing (default: None)

    Returns:
        List of execution times in nanoseconds, one per iteration
    """
    assert all(
        array.size > 0 for array in input_args + output_args
    ), "All NP arrays must have > 0 elements"

    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")

    logger = _get_logger()
    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    gpu_ptrs: Optional[List[Any]] = None
    padded_buffers: Optional[List[Any]] = None
    has_padding = False
    module = None
    function = None
    start_event = None
    stop_event = None

    actual_device_id = device_id if device_id is not None else hip_get_device()

    try:
        _log_with_device(
            logger,
            actual_device_id,
            f"Starting execution: kernel={kernel_name}, iterations={num_iterations}",
        )

        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()

        module = hip_module_load_data(hsaco_binary)
        function = hip_module_get_function(module, kernel_name.encode())
        _log_with_device(
            logger, actual_device_id, f"Loaded HSACO: {os.path.basename(hsaco_path)}"
        )

        all_arrays = input_args + output_args

        if padding_bytes is None:
            padding_bytes = [0] * len(all_arrays)
        elif len(padding_bytes) != len(all_arrays):
            raise ValueError(
                f"padding_bytes must have {len(all_arrays)} elements, got {len(padding_bytes)}"
            )

        has_padding = any(pb > 0 for pb in padding_bytes)

        if has_padding:
            padded_buffers = []
            ptr_values = []
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} padded buffers"
            )
            for arr, pad_bytes in zip(all_arrays, padding_bytes):
                base_ptr, data_ptr, _ = utils.copy_array_to_gpu(arr, pad_bytes)
                padded_buffers.append(base_ptr)
                ptr_values.append(utils.unwrap_pointer_from_capsule(data_ptr))
            params_tuple = utils.create_kernel_args_capsule(ptr_values)
        else:
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} buffers"
            )
            params_tuple, gpu_ptrs = utils.create_kernel_args_capsule_from_numpy(
                *all_arrays, device_id=device_id
            )

        iteration_times_ns = []
        start_event = hip_event_create()
        stop_event = hip_event_create()

        _log_with_device(
            logger,
            actual_device_id,
            f"Launching kernel: grid={grid_dim}, block={block_dim}",
        )

        if flush_llc is not None:
            flush_llc.initialize()

        for iteration in range(num_iterations):
            if flush_llc is not None:
                flush_llc.flush_llc()

            hip_event_record(start_event)
            hip_module_launch_kernel(
                function,
                grid_dim[0],
                grid_dim[1],
                grid_dim[2],
                block_dim[0],
                block_dim[1],
                block_dim[2],
                params_tuple[0],
            )
            hip_event_record(stop_event)
            hip_event_synchronize(stop_event)

            elapsed_ms = hip_event_elapsed_time(start_event, stop_event)
            elapsed_ns = int(elapsed_ms * 1_000_000)
            iteration_times_ns.append(elapsed_ns)

            _log_with_device(
                logger, actual_device_id, f"Iteration {iteration}: {elapsed_ms:.3f}ms"
            )

            if iteration == 0:
                _log_with_device(logger, actual_device_id, "Verifying results")
                num_inputs = len(input_args)
                if has_padding:
                    assert padded_buffers is not None
                    for i, output_arr in enumerate(output_args):
                        utils.copy_from_gpu_buffer(
                            padded_buffers[num_inputs + i],
                            output_arr,
                            padding_bytes[num_inputs + i],
                        )
                else:
                    assert gpu_ptrs is not None
                    for i, output_arr in enumerate(output_args):
                        capsule_output = utils.wrap_pointer_in_capsule(
                            output_arr.ctypes.data
                        )
                        hip_memcpy_device_to_host(
                            capsule_output, gpu_ptrs[num_inputs + i], output_arr.nbytes
                        )

                if verify_fn is not None:
                    verify_fn(input_args, output_args)
                    _log_with_device(logger, actual_device_id, "Verification passed")

        avg_time_ms = sum(iteration_times_ns) / len(iteration_times_ns) / 1_000_000
        _log_with_device(
            logger,
            actual_device_id,
            f"Completed {num_iterations} iterations, avg={avg_time_ms:.3f}ms",
        )

        return iteration_times_ns

    finally:
        if start_event is not None:
            hip_event_destroy(start_event)
        if stop_event is not None:
            hip_event_destroy(stop_event)
        if padded_buffers is not None:
            for ptr in padded_buffers:
                hip_free(ptr)
        elif gpu_ptrs is not None:
            for ptr in gpu_ptrs:
                hip_free(ptr)
        if flush_llc is not None:
            flush_llc.cleanup()
        if function is not None:
            hip_function_free(function)
        if module is not None:
            hip_module_unload(module)
