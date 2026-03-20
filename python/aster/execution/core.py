"""MLIR/LLVM-free HIP execution utilities.

This module merges capsule wrapping, GPU memory management, and kernel
execution. It does NOT import any MLIR or LLVM libraries, making it safe
to use under rocprofv3 (which crashes when LLVM.so is loaded alongside its
own LLVM).

Usage:
    from aster.execution.core import execute_hsaco
    from aster.execution.utils import system_has_gpu

    if not system_has_gpu("gfx942"):
        pytest.skip("no GPU")

    times_ns = execute_hsaco(
        hsaco_path="kernel.hsaco",
        kernel_name="my_kernel",
        input_arrays=[A, B],
        output_arrays=[C],
        grid_dim=(304, 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=5,
    )
"""

import ctypes
from typing import Any, Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# PyCapsule helpers (private + public aliases)
# ---------------------------------------------------------------------------


def _capsule(ptr):
    """Wrap a raw pointer in a PyCapsule (nb_handle)."""
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ptr, b"nb_handle", None)


def _uncapsule(capsule):
    """Extract a raw pointer from a PyCapsule (nb_handle)."""
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return PyCapsule_GetPointer(ctypes.py_object(capsule), b"nb_handle")


# Public aliases used by aster.utils and external callers.
wrap_pointer_in_capsule = _capsule
unwrap_pointer_from_capsule = _uncapsule


# ---------------------------------------------------------------------------
# Scalar detection
# ---------------------------------------------------------------------------


def _is_scalar(arg: Any) -> bool:
    """Return true if arg is a Python or numpy scalar (passed by value to the kernel)."""
    import numpy as np

    return isinstance(arg, (int, float, np.integer, np.floating))


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------


def copy_array_to_gpu(array):
    """Copy a numpy array to GPU memory.

    Args:
        array: numpy array to copy.

    Returns:
        Tuple of (gpu_ptr, ptr_value) where:
        - gpu_ptr: GPU pointer to the allocated buffer (for freeing and kernel args).
        - ptr_value: Raw pointer value of gpu_ptr.
    """
    import numpy as np
    from aster._mlir_libs._runtime_module import hip_malloc, hip_memcpy_host_to_device

    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array)}")

    gpu_ptr = hip_malloc(array.nbytes)
    if gpu_ptr is None:
        raise RuntimeError(
            f"Failed to allocate GPU memory of size {array.nbytes} for shape {array.shape}"
        )

    ptr_value = _uncapsule(gpu_ptr)
    hip_memcpy_host_to_device(gpu_ptr, _capsule(array.ctypes.data), array.nbytes)

    return gpu_ptr, ptr_value


def copy_from_gpu_buffer(gpu_ptr, host_array):
    """Copy data from GPU buffer to host array.

    Args:
        gpu_ptr: GPU pointer to the buffer.
        host_array: Host numpy array to copy data into.
    """
    from aster._mlir_libs._runtime_module import hip_memcpy_device_to_host

    hip_memcpy_device_to_host(
        _capsule(host_array.ctypes.data), gpu_ptr, host_array.nbytes
    )


# ---------------------------------------------------------------------------
# Kernel argument capsule construction
# ---------------------------------------------------------------------------


def create_kernel_args_capsule(ptr_values):
    """Create kernel arguments capsule from pointer values.

    Args:
        ptr_values: List of raw pointer values.

    Returns:
        Tuple of (args_capsule, kernel_args, kernel_ptr_arr) where:
        - args_capsule: PyCapsule for hip_module_launch_kernel.
        - kernel_args: ctypes structure containing the arguments.
        - kernel_ptr_arr: Array of pointers to the structure fields.
    """

    class _Args(ctypes.Structure):
        _fields_ = [(f"_field{i}", ctypes.c_void_p) for i in range(len(ptr_values))]

    kernel_args = _Args()
    for i, ptr_val in enumerate(ptr_values):
        setattr(kernel_args, f"_field{i}", ptr_val)

    ptr_arr_t = ctypes.c_void_p * len(ptr_values)
    kernel_args_addr = ctypes.addressof(kernel_args)
    kernel_ptr_arr = ptr_arr_t(
        *[
            kernel_args_addr + getattr(_Args, f"_field{i}").offset
            for i in range(len(ptr_values))
        ]
    )

    args_capsule = _capsule(ctypes.addressof(kernel_ptr_arr))
    return args_capsule, kernel_args, kernel_ptr_arr


def create_kernel_args_capsule_from_numpy(*arrays, device_id: Optional[int] = None):
    """Create a kernel arguments capsule from numpy arrays for HIP kernel launch.

    Args:
        *arrays: Variable number of numpy arrays to pass as kernel arguments.

    Returns:
        A tuple of (params_tuple, gpu_ptrs) where:
        - params_tuple: Tuple of (args_capsule, kernel_args, kernel_ptr_arr).
        - gpu_ptrs: List of GPU pointers that should be freed after kernel execution.
    """
    assert all(array.size > 0 for array in arrays), "All arrays must have > 0 elements"

    gpu_ptrs = []
    ptr_values = []
    for array in arrays:
        gpu_ptr, ptr_value = copy_array_to_gpu(array)
        gpu_ptrs.append(gpu_ptr)
        ptr_values.append(ptr_value)

    args_capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(ptr_values)
    return (args_capsule, kernel_args, kernel_ptr_arr), gpu_ptrs


# ---------------------------------------------------------------------------
# Internal kernel launch helpers
# ---------------------------------------------------------------------------


def _prepare_kernel_args(
    args: List[Any],
) -> Tuple[List[Any], Tuple[Any, Any], Any]:
    """Allocate GPU buffers for array args, copy host data, and build the kernel args struct.

    Scalars are packed directly as c_void_p values (by-value passing). Arrays are
    copied to freshly allocated GPU memory.

    Args:
        args: Flat list of kernel arguments (scalars or numpy arrays).

    Returns:
        A (gpu_ptrs, keep_alive, args_capsule) tuple where:
        - gpu_ptrs: Base GPU allocations to free after the kernel finishes.
        - keep_alive: (kernel_args, kernel_ptr_arr) ctypes objects that must
          remain alive until hip_module_launch_kernel returns.
        - args_capsule: PyCapsule to pass to hip_module_launch_kernel.
    """
    from aster._mlir_libs._runtime_module import hip_malloc, hip_memcpy_host_to_device

    gpu_ptrs: List[Any] = []
    ptr_values: List[int] = []

    for arg in args:
        if _is_scalar(arg):
            ptr_values.append(int(arg))
            continue
        gpu_ptr = hip_malloc(arg.nbytes)
        gpu_ptrs.append(gpu_ptr)
        base_val = _uncapsule(gpu_ptr)
        hip_memcpy_host_to_device(
            _capsule(base_val), _capsule(arg.ctypes.data), arg.nbytes
        )
        ptr_values.append(base_val)

    class _Args(ctypes.Structure):
        _fields_ = [(f"f{i}", ctypes.c_void_p) for i in range(len(ptr_values))]

    kernel_args = _Args()
    for i, pv in enumerate(ptr_values):
        setattr(kernel_args, f"f{i}", pv)
    ptr_arr_t = ctypes.c_void_p * len(ptr_values)
    ka_addr = ctypes.addressof(kernel_args)
    kernel_ptr_arr = ptr_arr_t(
        *[ka_addr + getattr(_Args, f"f{i}").offset for i in range(len(ptr_values))]
    )
    return (
        gpu_ptrs,
        (kernel_args, kernel_ptr_arr),
        _capsule(ctypes.addressof(kernel_ptr_arr)),
    )


class _TimedLaunch:
    """Context manager that brackets a GPU kernel launch with HIP event timing.

    Records start/stop events around the ``with`` body, synchronizes on exit,
    and exposes the wall time as ``elapsed_ns``.

    Args:
        start_event: Pre-created HIP event used as the start marker.
        stop_event: Pre-created HIP event used as the stop marker.
        flush_llc: Optional LLC-flush object. ``flush_llc()`` is called before
            the start event is recorded if provided.

    Example::

        with _TimedLaunch(start_event, stop_event, flush_llc) as t:
            hip_module_launch_kernel(function, *grid, *block, args_capsule)
        times_ns.append(t.elapsed_ns)
    """

    elapsed_ns: int

    def __init__(self, start_event, stop_event, flush_llc=None):
        self._start = start_event
        self._stop = stop_event
        self._flush_llc = flush_llc

    def __enter__(self):
        from aster._mlir_libs._runtime_module import hip_event_record

        if self._flush_llc is not None:
            self._flush_llc.flush_llc()
        hip_event_record(self._start)
        return self

    def __exit__(self, *_):
        from aster._mlir_libs._runtime_module import (
            hip_event_elapsed_time,
            hip_event_record,
            hip_event_synchronize,
        )

        hip_event_record(self._stop)
        hip_event_synchronize(self._stop)
        self.elapsed_ns = int(
            hip_event_elapsed_time(self._start, self._stop) * 1_000_000
        )


def _copy_outputs_from_gpu(
    gpu_ptrs: List[Any],
    input_arrays: List[Any],
    output_arrays: List[Any],
) -> None:
    """Copy output arrays back from GPU to host (in-place)."""
    from aster._mlir_libs._runtime_module import hip_memcpy_device_to_host

    gpu_i = sum(1 for a in input_arrays if not _is_scalar(a))
    for out_arr in output_arrays:
        if _is_scalar(out_arr):
            continue
        hip_memcpy_device_to_host(
            _capsule(out_arr.ctypes.data), gpu_ptrs[gpu_i], out_arr.nbytes
        )
        gpu_i += 1


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------


def execute_hsaco(
    hsaco_path: str,
    kernel_name: str,
    input_arrays: list,
    output_arrays: list,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
    verify_fn: Optional[Callable] = None,
) -> List[int]:
    """Execute a pre-compiled HSACO kernel on GPU.

    Args:
        hsaco_path: Path to the .hsaco file.
        kernel_name: Name of the kernel entry point.
        input_arrays: List of numpy arrays or scalars (read-only kernel inputs).
            Scalars (int/float) are passed by value; arrays are copied to GPU.
        output_arrays: List of numpy arrays (copied to GPU, modified in-place
            with results after the first iteration).
        grid_dim: (gridX, gridY, gridZ).
        block_dim: (blockX, blockY, blockZ).
        num_iterations: Number of kernel launches (for timing).
        device_id: GPU device to use. None keeps the current device.
        flush_llc: Optional object with initialize()/flush_llc()/cleanup()
            methods called around each iteration for LLC flushing.
        verify_fn: Optional callable invoked as verify_fn(input_arrays, output_arrays)
            after the first iteration to validate results.

    Returns:
        List of execution times in nanoseconds, one per iteration.
    """
    if hsaco_path is None:
        raise RuntimeError("hsaco_path is None — assembly failed")
    assert all(
        (_is_scalar(a) or a.size > 0) for a in input_arrays + output_arrays
    ), "all numpy arrays must have > 0 elements"

    # HIP-only imports (no MLIR/LLVM).
    from aster._mlir_libs._runtime_module import (
        hip_init,
        hip_set_device,
        hip_free,
        hip_module_load_data,
        hip_module_get_function,
        hip_module_launch_kernel,
        hip_module_unload,
        hip_function_free,
        hip_event_create,
        hip_event_destroy,
    )

    all_arrays = input_arrays + output_arrays
    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    with open(hsaco_path, "rb") as f:
        hsaco_binary = f.read()
    module = hip_module_load_data(hsaco_binary)
    function = hip_module_get_function(module, kernel_name.encode())

    gpu_ptrs, keep_alive, args_capsule = _prepare_kernel_args(all_arrays)

    start_event = hip_event_create()
    stop_event = hip_event_create()
    times_ns = []

    if flush_llc is not None:
        flush_llc.initialize()

    try:
        for it in range(num_iterations):
            with _TimedLaunch(start_event, stop_event, flush_llc) as t:
                hip_module_launch_kernel(
                    function,
                    grid_dim[0],
                    grid_dim[1],
                    grid_dim[2],
                    block_dim[0],
                    block_dim[1],
                    block_dim[2],
                    args_capsule,
                )
            times_ns.append(t.elapsed_ns)

            if it == 0:
                _copy_outputs_from_gpu(gpu_ptrs, input_arrays, output_arrays)
                if verify_fn is not None:
                    verify_fn(input_arrays, output_arrays)
    finally:
        if flush_llc is not None:
            flush_llc.cleanup()
        hip_event_destroy(start_event)
        hip_event_destroy(stop_event)
        for gp in gpu_ptrs:
            hip_free(gp)
        hip_function_free(function)
        hip_module_unload(module)

    return times_ns
