"""MLIR/LLVM-free HIP execution utilities.

This module provides GPU kernel execution using ONLY the HIP runtime .so
(aster._mlir_libs._runtime_module). It does NOT import any MLIR or LLVM
libraries, making it safe to use under rocprofv3 (which crashes when LLVM.so
is loaded alongside its own LLVM).

Usage:
    from aster.execution.hip import execute_hsaco, system_has_gpu, parse_asm_kernel_resources

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aster.core.target import Target


@dataclass
class KernelResources:
    """Resource usage extracted from AMDGPU assembly metadata.

    This is the ground truth for what the hardware will actually use, as emitted by the
    ASTER compiler in .amdgpu_metadata.
    """

    # Registers
    vgpr_count: int = 0
    sgpr_count: int = 0
    agpr_count: int = 0
    vgpr_spill_count: int = 0
    sgpr_spill_count: int = 0

    # Memory
    lds_bytes: int = 0  # .group_segment_fixed_size
    scratch_bytes: int = 0  # .private_segment_fixed_size
    kernarg_bytes: int = 0  # .kernarg_segment_size

    # Workgroup
    max_flat_workgroup_size: int = 0
    wavefront_size: int = 64

    @property
    def registers_str(self):
        """Compact register summary."""
        parts = [f"vgpr={self.vgpr_count}", f"sgpr={self.sgpr_count}"]
        if self.agpr_count > 0:
            parts.append(f"agpr={self.agpr_count}")
        if self.vgpr_spill_count > 0:
            parts.append(f"vgpr_spill={self.vgpr_spill_count}")
        if self.sgpr_spill_count > 0:
            parts.append(f"sgpr_spill={self.sgpr_spill_count}")
        return ", ".join(parts)

    def __str__(self):
        parts = [self.registers_str]
        parts.append(f"lds={self.lds_bytes}")
        if self.scratch_bytes > 0:
            parts.append(f"scratch={self.scratch_bytes}")
        return ", ".join(parts)

    def check_occupancy(self, num_threads: int, mcpu: str = "gfx942") -> List[str]:
        """Return a list of occupancy violations for the given target.

        Returns an empty list if the kernel can launch or the mcpu is unknown.
        """
        try:
            target = Target.from_mcpu(mcpu)
        except ValueError:
            return []
        num_waves = (num_threads + target.wavefront_size - 1) // target.wavefront_size
        waves_per_simd = (num_waves + target.num_simds - 1) // target.num_simds
        violations = []

        if self.vgpr_count > target.max_vgprs:
            violations.append(f"vgpr per thread {self.vgpr_count} > {target.max_vgprs}")
        if target.max_agprs > 0 and self.agpr_count > target.max_agprs:
            violations.append(f"agpr per thread {self.agpr_count} > {target.max_agprs}")
        combined = self.vgpr_count + self.agpr_count
        combined_max = target.max_vgprs + target.max_agprs
        if combined > combined_max:
            violations.append(
                f"(vgpr+agpr) per wave {self.vgpr_count}+{self.agpr_count}"
                f"={combined} > {combined_max}"
            )
        vgpr_simd = self.vgpr_count * waves_per_simd
        if vgpr_simd > target.vgprs_per_simd:
            violations.append(
                f"vgpr per SIMD {self.vgpr_count}*{waves_per_simd}"
                f"={vgpr_simd} > {target.vgprs_per_simd}"
            )
        if target.agprs_per_simd > 0:
            agpr_simd = self.agpr_count * waves_per_simd
            if agpr_simd > target.agprs_per_simd:
                violations.append(
                    f"agpr per SIMD {self.agpr_count}*{waves_per_simd}"
                    f"={agpr_simd} > {target.agprs_per_simd}"
                )
        if self.lds_bytes > target.lds_per_cu:
            violations.append(
                f"LDS per workgroup {self.lds_bytes} > {target.lds_per_cu}"
            )
        return violations


def compute_register_budget(
    num_threads: int,
    mcpu: str = "gfx942",
    num_wg_per_cu: int = 1,
) -> Tuple[int, int, int]:
    """Compute per-wave VGPR/AGPR limits and per-WG LDS limit from occupancy target.

    Args:
        num_threads: Number of threads per workgroup.
        mcpu: Target GPU (e.g. "gfx942", "gfx950").
        num_wg_per_cu: Number of workgroups sharing a CU (default 1).

    Returns:
        (max_vgprs, max_agprs, lds_per_wg) tuple.
    """
    try:
        target = Target.from_mcpu(mcpu)
    except ValueError:
        target = Target.from_mcpu("gfx942")
    num_waves = (num_threads + target.wavefront_size - 1) // target.wavefront_size
    total_waves = num_waves * num_wg_per_cu
    waves_per_simd = (total_waves + target.num_simds - 1) // target.num_simds
    max_vgprs = min(target.max_vgprs, target.vgprs_per_simd // waves_per_simd)
    max_agprs = min(target.max_agprs, target.agprs_per_simd // waves_per_simd)
    lds_per_wg = target.lds_per_cu // num_wg_per_cu
    return max_vgprs, max_agprs, lds_per_wg


# All integer fields we extract from .amdgpu_metadata YAML.
_METADATA_FIELDS = [
    (r"\.vgpr_count:\s*(\d+)", "vgpr_count"),
    (r"\.sgpr_count:\s*(\d+)", "sgpr_count"),
    (r"\.agpr_count:\s*(\d+)", "agpr_count"),
    (r"\.vgpr_spill_count:\s*(\d+)", "vgpr_spill_count"),
    (r"\.sgpr_spill_count:\s*(\d+)", "sgpr_spill_count"),
    (r"\.group_segment_fixed_size:\s*(\d+)", "lds_bytes"),
    (r"\.private_segment_fixed_size:\s*(\d+)", "scratch_bytes"),
    (r"\.kernarg_segment_size:\s*(\d+)", "kernarg_bytes"),
    (r"\.max_flat_workgroup_size:\s*(\d+)", "max_flat_workgroup_size"),
    (r"\.wavefront_size:\s*(\d+)", "wavefront_size"),
]


def _parse_metadata_yaml(
    meta_text: str, kernel_name: Optional[str] = None
) -> Dict[str, KernelResources]:
    """Parse the amdhsa.kernels YAML body into KernelResources.

    meta_text is everything between the --- delimiters in .amdgpu_metadata.
    """
    results = {}

    # Split into per-kernel blocks. Each kernel starts with "  - .agpr_count:" or
    # "  - .name:" -- we split on the "  - ." pattern that starts a new kernel entry.
    kernel_blocks = re.split(r"\n  - \.", meta_text)
    # First element is "amdhsa.kernels:\n" header, skip it.

    for i, block in enumerate(kernel_blocks):
        if i == 0:
            continue
        # Re-add the leading "." that was consumed by the split
        block = "." + block

        name_match = re.search(r"\.name:\s*(\S+)", block)
        if not name_match:
            continue
        name = name_match.group(1)

        if kernel_name is not None and name != kernel_name:
            continue

        res = KernelResources()
        for pattern, attr in _METADATA_FIELDS:
            m = re.search(pattern, block)
            if m:
                setattr(res, attr, int(m.group(1)))

        results[name] = res

    return results


def parse_asm_kernel_resources(
    asm: str, kernel_name: Optional[str] = None
) -> Dict[str, KernelResources]:
    """Parse kernel resource usage from AMDGPU assembly text.

    Extracts register counts, LDS size, scratch size, and workgroup limits
    from the .amdgpu_metadata YAML section emitted by the ASTER compiler.

    Args:
        asm: Assembly text containing .amdgpu_metadata section.
        kernel_name: If provided, only return resources for this kernel.
            If None, return resources for all kernels found.

    Returns:
        Dict mapping kernel name to KernelResources.
    """
    meta_match = re.search(
        r"\.amdgpu_metadata\s*\n---\s*\n(.*?)\n---\s*\n\s*\.end_amdgpu_metadata",
        asm,
        re.DOTALL,
    )
    if not meta_match:
        return {}
    return _parse_metadata_yaml(meta_match.group(1), kernel_name)


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


def _is_scalar(arg: Any) -> bool:
    """Return true if arg is a Python or numpy scalar (passed by value to the kernel)."""
    import numpy as np

    return isinstance(arg, (int, float, np.integer, np.floating))


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
        - gpu_ptrs: base GPU allocations to free after the kernel finishes.
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

    Returns:
        List of execution times in nanoseconds, one per iteration.
    """
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
