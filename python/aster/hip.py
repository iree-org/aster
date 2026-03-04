"""MLIR/LLVM-free HIP execution utilities.

This module provides GPU kernel execution using ONLY the HIP runtime .so
(aster._mlir_libs._runtime_module). It does NOT import any MLIR or LLVM
libraries, making it safe to use under rocprofv3 (which crashes when LLVM.so
is loaded alongside its own LLVM).

Usage:
    from aster.hip import execute_hsaco, system_has_gpu

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
import re
import subprocess
from typing import List, Optional, Tuple


def system_has_gpu(mcpu: str) -> bool:
    """Check if a GPU matching mcpu is available via rocminfo.

    Does NOT import aster/MLIR/LLVM. This is the single canonical
    implementation -- aster.utils.system_has_mcpu delegates here.
    """
    base_mcpu = mcpu.split(":")[0]
    import shutil

    rocminfo_path = shutil.which("rocminfo")
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        print(
            "WARNING: rocminfo not found on PATH. "
            "Install ROCm or add its bin/ directory to PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        print(f"WARNING: rocminfo timed out after 30s (path: {rocminfo_path}).")
        return False

    if result.returncode != 0:
        print(f"WARNING: rocminfo exited with code {result.returncode}.")
        return False

    raw_matches = re.findall(r"gfx[0-9]{3,4}[a-z0-9]*", result.stdout)
    archs = set(a.split(":")[0] for a in raw_matches)
    found = base_mcpu in archs
    if not found:
        print(
            f"DEBUG system_has_gpu: looking for '{base_mcpu}', "
            f"rocminfo found archs={sorted(archs)}, "
            f"raw_matches={raw_matches}"
        )
    return found


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


def execute_hsaco(
    hsaco_path: str,
    kernel_name: str,
    input_arrays: list,
    output_arrays: list,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    num_iterations: int = 1,
) -> Tuple[List[int]]:
    """Execute a pre-compiled HSACO kernel on GPU.

    Args:
        hsaco_path: Path to the .hsaco file.
        kernel_name: Name of the kernel entry point.
        input_arrays: List of numpy arrays (copied to GPU, read-only).
        output_arrays: List of numpy arrays (copied to GPU, results copied
            back after first iteration).
        grid_dim: (gridX, gridY, gridZ).
        block_dim: (blockX, blockY, blockZ).
        num_iterations: Number of kernel launches (for timing).

    Returns:
        List of execution times in nanoseconds, one per iteration.
        output_arrays are modified in-place with results from iteration 0.
    """
    import numpy as np

    # HIP-only imports (no MLIR/LLVM).
    from aster._mlir_libs._runtime_module import (
        hip_init,
        hip_malloc,
        hip_free,
        hip_memcpy_host_to_device,
        hip_memcpy_device_to_host,
        hip_module_load_data,
        hip_module_get_function,
        hip_module_launch_kernel,
        hip_module_unload,
        hip_function_free,
        hip_event_create,
        hip_event_destroy,
        hip_event_record,
        hip_event_synchronize,
        hip_event_elapsed_time,
    )

    all_arrays = [a.flatten() for a in input_arrays] + [
        a.flatten() for a in output_arrays
    ]
    num_inputs = len(input_arrays)

    hip_init()

    # Load HSACO.
    with open(hsaco_path, "rb") as f:
        hsaco_binary = f.read()
    module = hip_module_load_data(hsaco_binary)
    function = hip_module_get_function(module, kernel_name.encode())

    # Allocate GPU buffers and copy data.
    gpu_ptrs = []
    ptr_values = []
    for arr in all_arrays:
        gpu_ptr = hip_malloc(arr.nbytes)
        gpu_ptrs.append(gpu_ptr)
        ptr_val = _uncapsule(gpu_ptr)
        ptr_values.append(ptr_val)
        host_cap = _capsule(arr.ctypes.data)
        hip_memcpy_host_to_device(gpu_ptr, host_cap, arr.nbytes)

    # Build kernel args struct.
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
    args_capsule = _capsule(ctypes.addressof(kernel_ptr_arr))

    # Launch kernel.
    start_event = hip_event_create()
    stop_event = hip_event_create()
    times_ns = []
    try:
        for it in range(num_iterations):
            hip_event_record(start_event)
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
            hip_event_record(stop_event)
            hip_event_synchronize(stop_event)
            elapsed_ms = hip_event_elapsed_time(start_event, stop_event)
            times_ns.append(int(elapsed_ms * 1_000_000))

            # Copy outputs back on first iteration for correctness checks.
            if it == 0:
                for i, out_arr in enumerate(output_arrays):
                    flat = all_arrays[num_inputs + i]
                    out_cap = _capsule(flat.ctypes.data)
                    hip_memcpy_device_to_host(
                        out_cap, gpu_ptrs[num_inputs + i], flat.nbytes
                    )
                    np.copyto(out_arr.reshape(-1), flat)
    finally:
        hip_event_destroy(start_event)
        hip_event_destroy(stop_event)
        for gp in gpu_ptrs:
            hip_free(gp)
        hip_function_free(function)
        hip_module_unload(module)

    return times_ns
