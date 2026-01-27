################################################################################
# MLIR utils.
################################################################################
from pathlib import Path
from typing import List, Optional


def translate_module(module, debug_print=False):
    """Translate an AMDGCN module to assembly.

    Args:
        module: The AMDGCN module to translate.
        debug_print: If True, print debug comments for AllocaOp and MakeRegisterRangeOp.

    Returns:
        The assembly string representation of the module.
    """
    from aster._mlir_libs._amdgcn import translate_module as _translate_module

    return _translate_module(module.operation, debug_print)


################################################################################
# Capsule / nanobind utils.
################################################################################
def wrap_pointer_in_capsule(ptr):
    """Wrap a pointer in a PyCapsule.

    Args:
        ptr: ctypes pointer value (c_void_p or ctypes.addressof result)

    Returns:
        PyCapsule containing the pointer
    """
    import ctypes
    from ctypes import pythonapi, py_object, c_void_p, c_char_p

    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
    return PyCapsule_New(ptr, b"nb_handle", None)


def unwrap_pointer_from_capsule(capsule):
    """Extract a pointer value from a PyCapsule.

    Args:
        capsule: PyCapsule containing a pointer

    Returns:
        Raw pointer value (c_void_p)
    """
    import ctypes

    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return PyCapsule_GetPointer(ctypes.py_object(capsule), b"nb_handle")


def copy_array_to_gpu(array, padding_bytes: int = 0):
    """Copy a numpy array to GPU memory with optional padding.

    Args:
        array: numpy array to copy
        padding_bytes: Number of padding bytes to add before the data (default: 0)

    Returns:
        Tuple of (base_gpu_ptr, data_gpu_ptr, base_ptr_value) where:
        - base_gpu_ptr: Base pointer to the entire allocated buffer (for freeing)
        - data_gpu_ptr: Pointer to the data location within the buffer (for kernel args)
        - base_ptr_value: Raw pointer value of base_gpu_ptr
    """
    from aster._mlir_libs._runtime_module import (
        hip_malloc,
        hip_memcpy_host_to_device,
    )

    import numpy as np

    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array)}")

    # Allocate GPU memory (with padding if specified)
    total_size = array.nbytes + padding_bytes
    base_gpu_ptr = hip_malloc(total_size)
    if base_gpu_ptr is None:
        raise RuntimeError(
            f"Failed to allocate GPU memory of size {total_size} for shape {array.shape}"
        )

    # Calculate pointer to data location (with padding offset)
    base_ptr_value = unwrap_pointer_from_capsule(base_gpu_ptr)
    data_ptr_value = base_ptr_value + padding_bytes

    # Wrap data pointer in capsule for kernel arguments
    data_gpu_ptr = wrap_pointer_in_capsule(data_ptr_value)

    # Wrap host pointer and copy data to GPU at offset location
    host_capsule = wrap_pointer_in_capsule(array.ctypes.data)
    hip_memcpy_host_to_device(data_gpu_ptr, host_capsule, array.nbytes)

    return base_gpu_ptr, data_gpu_ptr, base_ptr_value


def copy_from_gpu_buffer(base_gpu_ptr, host_array, padding_bytes: int = 0):
    """Copy data from GPU buffer to host array.

    Args:
        base_gpu_ptr: Base pointer to the GPU buffer
        host_array: Host numpy array to copy data into
        padding_bytes: Number of padding bytes before the data (default: 0)
    """
    from aster._mlir_libs._runtime_module import hip_memcpy_device_to_host

    base_ptr_value = unwrap_pointer_from_capsule(base_gpu_ptr)
    data_ptr_value = base_ptr_value + padding_bytes

    data_gpu_ptr = wrap_pointer_in_capsule(data_ptr_value)
    host_capsule = wrap_pointer_in_capsule(host_array.ctypes.data)

    hip_memcpy_device_to_host(host_capsule, data_gpu_ptr, host_array.nbytes)


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
        base_gpu_ptr, _, ptr_value = copy_array_to_gpu(array)
        gpu_ptrs.append(base_gpu_ptr)
        ptr_values.append(ptr_value)

    # Step 2: Create kernel arguments capsule
    args_capsule, kernel_args, kernel_ptr_arr = create_kernel_args_capsule(ptr_values)

    # Keep references to prevent garbage collection
    return (args_capsule, kernel_args, kernel_ptr_arr), gpu_ptrs


################################################################################
# Runtime CLI utils.
################################################################################
def _detect_gfx_archs_via_rocminfo(rocm_path: Optional[Path]) -> List[str]:
    import subprocess
    import re

    archs = set()
    candidates = []
    if rocm_path is not None:
        candidates.append(rocm_path / "bin" / "rocminfo")
    candidates.append(Path("rocminfo"))

    for exe in candidates:
        try:
            result = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                continue
            found = re.findall(r"gfx[0-9]{3,4}[a-z0-9]*", result.stdout)
            archs.update(a.split(":")[0] for a in found)
            if archs:
                break
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue
    return sorted(archs)


def system_has_mcpu(mcpu: str, rocm_path: Optional[Path] = None) -> bool:
    archs = set()
    for arch in _detect_gfx_archs_via_rocminfo(rocm_path):
        archs.add(arch)
    base_mcpu = mcpu.split(":")[0]
    return base_mcpu in archs


def compile_to_hsaco(
    asm_content, target="gfx942", wavefront_size=64
) -> Optional[bytes]:
    """Compile AMDGPU assembly to hsaco binary."""
    from aster._mlir_libs._amdgcn import compile_asm as _compile_asm
    from aster.ir import Location, Context

    with Context() as ctx:
        return _compile_asm(
            Location.unknown(), asm_content, target, f"+wavefrontsize{wavefront_size}"
        )


def assemble_to_hsaco(
    asm_content, target="gfx942", wavefront_size=64, output_path: Optional[str] = None
) -> Optional[str]:
    """Assemble AMDGPU assembly to hsaco file.

    Args:
        asm_content: The assembly string to assemble.
        target: The AMDGPU target (e.g., "gfx942").
        wavefront_size: Wavefront size (32 or 64).
        output_path: Optional path to output hsaco file. If None, a temporary file is created.

    Returns:
        Path to the generated hsaco file.
    """
    hsaco_data = compile_to_hsaco(asm_content, target, wavefront_size)
    if hsaco_data is None:
        return None

    if output_path is None:
        import tempfile

        hsaco_file = tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False)
        output_path = hsaco_file.name
        hsaco_file.close()

    # Write the hsaco data to the file
    with open(output_path, "wb") as f:
        f.write(hsaco_data)

    return output_path
