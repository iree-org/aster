"""Common utilities for integration tests."""

import os
import numpy as np
from typing import Tuple, Callable, Optional, List

from aster import utils
from aster._mlir_libs._runtime_module import (
    hip_module_load_data,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_device_synchronize,
    hip_free,
    hip_memcpy_device_to_host,
    hip_module_unload,
    hip_function_free,
)


def execute_kernel_and_verify(
    asm_code: str,
    kernel_name: str,
    input_args: List[np.ndarray],
    output_args: List[np.ndarray],
    mcpu: str,
    wavefront_size: int = 32,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    verify_fn: Optional[Callable[[List[np.ndarray], List[np.ndarray]], None]] = None,
) -> None:
    """Execute a GPU kernel and verify its results.

    Args:
        asm_code: Assembly code to assemble
        kernel_name: Name of the kernel function
        input_args: List of input numpy arrays
        output_args: List of output numpy arrays (will be modified in-place)
        mcpu: Target GPU architecture (e.g., "gfx942", "gfx1201")
        wavefront_size: Wavefront size (default: 32)
        grid_dim: Grid dimensions (default: (1, 1, 1))
        block_dim: Block dimensions (default: (64, 1, 1))
        verify_fn: Custom verification function that takes (input_args, output_args).
    """
    # Assemble to hsaco
    hsaco_path = utils.assemble_to_hsaco(
        asm_code, target=mcpu, wavefront_size=wavefront_size
    )

    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")

    gpu_ptrs = None
    module = None
    function = None
    try:
        # Load hsaco binary
        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()

        module = hip_module_load_data(hsaco_binary)
        function = hip_module_get_function(module, kernel_name.encode())

        # Create kernel arguments from all input and output arrays
        all_arrays = input_args + output_args
        params_tuple, gpu_ptrs = utils.create_kernel_args_capsule_from_numpy(
            *all_arrays
        )

        # Launch kernel
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
        hip_device_synchronize()

        # Copy results back from all output arrays
        num_inputs = len(input_args)
        for i, output_arr in enumerate(output_args):
            output_ptr = gpu_ptrs[num_inputs + i]
            capsule_output = utils.wrap_pointer_in_capsule(output_arr.ctypes.data)
            hip_memcpy_device_to_host(capsule_output, output_ptr, output_arr.nbytes)

        # Verify results
        assert verify_fn, "No verify function provided"
        verify_fn(input_args, output_args)

    finally:
        # Cleanup
        if gpu_ptrs is not None:
            for ptr in gpu_ptrs:
                hip_free(ptr)
        if function is not None:
            hip_function_free(function)
        if module is not None:
            hip_module_unload(module)
        if hsaco_path and os.path.exists(hsaco_path):
            os.unlink(hsaco_path)
