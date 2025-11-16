"""Runtime utilities for executing AMDGCN kernels."""

from pathlib import Path
from typing import Optional

import numpy as np

from aster import utils


def execute_kernel_from_hsaco_file(
    hsaco_file_path: str,
    kernel_name: str,
    num_workgroups: int,
    num_wavefronts: int,
    wavefront_size: int,
    *,
    input_buffers: Optional[list[np.ndarray]] = None,
    output_buffers: Optional[list[np.ndarray]] = None,
    num_iterations: int = 1,
):
    """Load and execute a kernel from an existing .hsaco file.

    Args:
        hsaco_file_path: Path to the .hsaco file
        kernel_name: Name of the kernel function
        num_workgroups: Number of workgroups in x dimension
        num_wavefronts: Number of wavefronts per workgroup
        wavefront_size: Size of each wavefront
        input_buffers: Optional list of numpy arrays to pass as input buffers
        output_buffers: Optional list of numpy arrays to receive output data
        num_iterations: Number of times to launch the kernel (default: 1)
    """
    hsaco_path = Path(hsaco_file_path)

    if not hsaco_path.exists():
        raise FileNotFoundError(f"HSACO file {hsaco_file_path} does not exist")

    with open(hsaco_path, "rb") as f:
        binary = f.read()

    from aster._mlir_libs._runtime_module import (
        hip_device_synchronize,
        hip_free,
        hip_memcpy_device_to_host,
        hip_module_get_function,
        hip_module_launch_kernel,
        hip_module_load_data,
        hip_module_unload,
        hip_function_free,
    )

    m = None
    f = None
    try:
        m = hip_module_load_data(binary)
    except Exception as e:
        raise RuntimeError(f"Failed to load HIP module from {hsaco_file_path}: {e}")

    try:
        f = hip_module_get_function(m, kernel_name.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to get kernel function '{kernel_name}' from module: {e}"
        )

    # Allocate GPU memory for buffers if provided
    gpu_ptrs_tuples = []
    try:
        kernel_params_data = None
        if input_buffers or output_buffers:
            all_buffers = (input_buffers or []) + (output_buffers or [])

            try:
                # Use the function from aster.utils to create the capsule
                params_tuple, gpu_ptrs = utils.create_kernel_args_capsule_from_numpy(
                    *all_buffers
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to allocate GPU memory for kernel buffers: {e}"
                )

            # Store tuples for cleanup and copying back
            for i, buf in enumerate(all_buffers):
                gpu_ptrs_tuples.append((gpu_ptrs[i], buf, buf.nbytes))

            # Keep references to prevent garbage collection
            kernel_params_data = params_tuple

        print("\n")
        print(f"Launching kernel {num_iterations} time(s)")

        for iteration in range(num_iterations):
            try:
                hip_module_launch_kernel(
                    f,
                    num_workgroups,
                    1,
                    1,
                    num_wavefronts * wavefront_size,
                    1,
                    1,
                    kernel_params_data[0] if kernel_params_data else None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to launch kernel '{kernel_name}' (iteration {iteration + 1}/{num_iterations}) "
                    f"(workgroups={num_workgroups}, wavefronts={num_wavefronts}, "
                    f"wavefront_size={wavefront_size}): {e}"
                )

        try:
            hip_device_synchronize()
        except Exception as e:
            raise RuntimeError(f"Failed to synchronize device after kernel launch: {e}")

        # Copy output buffers back to host
        if output_buffers:
            for i, ((gpu_ptr, buf, size_bytes), out_buf) in enumerate(
                zip(gpu_ptrs_tuples[len(input_buffers or []) :], output_buffers)
            ):
                ptr_host = out_buf.ctypes.data
                from ctypes import pythonapi, py_object, c_void_p, c_char_p

                PyCapsule_New = pythonapi.PyCapsule_New
                PyCapsule_New.restype = py_object
                PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
                capsule_host = PyCapsule_New(ptr_host, b"nb_handle", None)

                try:
                    hip_memcpy_device_to_host(capsule_host, gpu_ptr, size_bytes)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to copy output buffer {i} from device to host: {e}"
                    )

        print("Done")

    finally:
        # Free GPU memory
        for i, (gpu_ptr, _, _) in enumerate(gpu_ptrs_tuples):
            try:
                hip_free(gpu_ptr)
            except Exception as e:
                print(f"Warning: Failed to free GPU memory for buffer {i}: {e}")
        # Cleanup module and function handles
        if f is not None:
            try:
                hip_function_free(f)
            except Exception as e:
                print(f"Warning: Failed to free function handle: {e}")
        if m is not None:
            try:
                hip_module_unload(m)
            except Exception as e:
                print(f"Warning: Failed to unload module: {e}")
