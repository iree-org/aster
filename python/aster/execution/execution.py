"""GPU memory management and kernel argument utilities."""

from typing import List, Optional

from aster.utils.capsule import wrap_pointer_in_capsule, unwrap_pointer_from_capsule


def copy_array_to_gpu(array):
    """Copy a numpy array to GPU memory.

    Args:
        array: numpy array to copy

    Returns:
        Tuple of (gpu_ptr, ptr_value) where:
        - gpu_ptr: GPU pointer to the allocated buffer (for freeing and kernel args)
        - ptr_value: Raw pointer value of gpu_ptr
    """
    from aster._mlir_libs._runtime_module import (
        hip_malloc,
        hip_memcpy_host_to_device,
    )

    import numpy as np

    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array)}")

    gpu_ptr = hip_malloc(array.nbytes)
    if gpu_ptr is None:
        raise RuntimeError(
            f"Failed to allocate GPU memory of size {array.nbytes} for shape {array.shape}"
        )

    ptr_value = unwrap_pointer_from_capsule(gpu_ptr)
    host_capsule = wrap_pointer_in_capsule(array.ctypes.data)
    hip_memcpy_host_to_device(gpu_ptr, host_capsule, array.nbytes)

    return gpu_ptr, ptr_value


def copy_from_gpu_buffer(gpu_ptr, host_array):
    """Copy data from GPU buffer to host array.

    Args:
        gpu_ptr: GPU pointer to the buffer
        host_array: Host numpy array to copy data into
    """
    from aster._mlir_libs._runtime_module import hip_memcpy_device_to_host

    host_capsule = wrap_pointer_in_capsule(host_array.ctypes.data)
    hip_memcpy_device_to_host(host_capsule, gpu_ptr, host_array.nbytes)


def create_kernel_args_capsule(ptr_values):
    """Create kernel arguments capsule from pointer values.

    Args:
        ptr_values: List of raw pointer values

    Returns:
        Tuple of (args_capsule, kernel_args, kernel_ptr_arr) where:
        - args_capsule: PyCapsule for hip_module_launch_kernel
        - kernel_args: ctypes structure containing the arguments
        - kernel_ptr_arr: Array of pointers to the structure fields
    """
    import ctypes

    # Create kernel arguments structure
    class _Args(ctypes.Structure):
        _fields_ = [(f"_field{i}", ctypes.c_void_p) for i in range(len(ptr_values))]

    kernel_args = _Args()
    for i, ptr_val in enumerate(ptr_values):
        setattr(kernel_args, f"_field{i}", ptr_val)

    # Create an array where each element is the address of a field in the structure
    ptr_arr_t = ctypes.c_void_p * len(ptr_values)
    kernel_args_addr = ctypes.addressof(kernel_args)
    kernel_ptr_arr = ptr_arr_t(
        *[
            kernel_args_addr + getattr(_Args, f"_field{i}").offset
            for i in range(len(ptr_values))
        ]
    )

    # Wrap the pointer array in a capsule
    args_capsule = wrap_pointer_in_capsule(ctypes.addressof(kernel_ptr_arr))

    return args_capsule, kernel_args, kernel_ptr_arr


def create_kernel_args_capsule_from_numpy(*arrays, device_id: Optional[int] = None):
    """Create a kernel arguments capsule from numpy arrays for HIP kernel launch.

    Args:
        *arrays: Variable number of numpy arrays to pass as kernel arguments

    Returns:
        A tuple of (params_tuple, gpu_ptrs) where:
        - params_tuple: Tuple of (args_capsule, kernel_args, kernel_ptr_arr)
          where args_capsule is the PyCapsule for hip_module_launch_kernel
        - gpu_ptrs: List of GPU pointers that should be freed after kernel execution

    Example:
        import numpy as np
        from aster._mlir_libs._runtime_module import (
            hip_module_load_data, hip_module_get_function,
            hip_module_launch_kernel, hip_device_synchronize, hip_free,
            hip_module_unload, hip_function_free
        )

        data1 = np.array([1, 2, 3], dtype=np.int32)
        data2 = np.array([4, 5, 6], dtype=np.int32)
        params_tuple, gpu_ptrs = create_kernel_args_capsule_from_numpy(data1, data2)

        # Load and launch kernel
        m = hip_module_load_data(hsaco_binary)
        f = hip_module_get_function(m, b"kernel_name")
        hip_module_launch_kernel(f, 1, 1, 1, 64, 1, 1, params_tuple[0])
        hip_device_synchronize()

        # Free GPU memory
        for ptr in gpu_ptrs:
            hip_free(ptr)
        # Cleanup module and function handles
        hip_function_free(f)
        hip_module_unload(m)
    """

    assert all(array.size > 0 for array in arrays), "All arrays must have > 0 elements"

    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required to use create_kernel_args_capsule_from_numpy"
        )

    # Step 1: Copy all arrays to GPU
    gpu_ptrs = []
    ptr_values = []
    for array in arrays:
        gpu_ptr, ptr_value = copy_array_to_gpu(array)
        gpu_ptrs.append(gpu_ptr)
        ptr_values.append(ptr_value)

    # Step 2: Create kernel arguments capsule
    args_capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(ptr_values)

    # Keep references to prevent garbage collection
    return (args_capsule, kernel_args, kernel_ptr_arr), gpu_ptrs
