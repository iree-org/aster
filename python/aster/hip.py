"""MLIR/LLVM-free HIP execution utilities.

This module provides GPU kernel execution using ONLY the HIP runtime .so
(aster._mlir_libs._runtime_module). It does NOT import any MLIR or LLVM
libraries, making it safe to use under rocprofv3 (which crashes when LLVM.so
is loaded alongside its own LLVM).

Usage:
    from aster.hip import execute_hsaco, system_has_gpu, parse_asm_kernel_resources

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
from typing import Dict, List, Optional, Tuple


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

        Returns an empty list if the kernel can launch.
        """
        if mcpu in ("gfx940", "gfx942"):
            return self._check_occupancy_cdna3(num_threads)
        if mcpu in ("gfx950",):
            return self._check_occupancy_cdna4(num_threads)
        return []

    def _check_occupancy_cdna3(
        self, num_threads: int, lds_per_cu: int = 65536
    ) -> List[str]:
        """CDNA3 occupancy rules (also used as base for CDNA4 with overrides).

        Register rules (per wave, ISA 3.6.4):
          - Per thread:  vgpr <= 256,  agpr <= 256
          - Per wave:    vgpr + agpr <= 512
        Per-SIMD file limits (512 VGPR + 512 AGPR per SIMD, 4 SIMDs per CU):
          - vgpr * waves_per_simd <= 512
          - agpr * waves_per_simd <= 512
        LDS:
          - lds_bytes <= lds_per_cu
        """
        _NUM_SIMDS = 4
        num_waves = (num_threads + self.wavefront_size - 1) // self.wavefront_size
        max_waves_per_simd = (num_waves + _NUM_SIMDS - 1) // _NUM_SIMDS
        violations = []

        if self.vgpr_count > 256:
            violations.append(f"vgpr per thread {self.vgpr_count} > 256")
        if self.agpr_count > 256:
            violations.append(f"agpr per thread {self.agpr_count} > 256")
        total_wave = self.vgpr_count + self.agpr_count
        if total_wave > 512:
            violations.append(
                f"(vgpr+agpr) per wave {self.vgpr_count}+{self.agpr_count}={total_wave} > 512"
            )
        vgpr_simd = self.vgpr_count * max_waves_per_simd
        if vgpr_simd > 512:
            violations.append(
                f"vgpr per SIMD {self.vgpr_count}*{max_waves_per_simd}={vgpr_simd} > 512"
            )
        agpr_simd = self.agpr_count * max_waves_per_simd
        if agpr_simd > 512:
            violations.append(
                f"agpr per SIMD {self.agpr_count}*{max_waves_per_simd}={agpr_simd} > 512"
            )
        if self.lds_bytes > lds_per_cu:
            violations.append(f"LDS per workgroup {self.lds_bytes} > {lds_per_cu}")
        return violations

    def _check_occupancy_cdna4(self, num_threads: int) -> List[str]:
        """Gfx950 (CDNA4) occupancy rules.

        Uses CDNA3 rules with CDNA4 overrides:
          - LDS: 256KB per CU (vs 64KB on CDNA3)
          - TODO: VGPR/AGPR per-thread limits for CDNA4
          - TODO: SIMD count / waves-per-SIMD limits for CDNA4
          - TODO: per-workgroup register budget for CDNA4
        """
        return self._check_occupancy_cdna3(num_threads, lds_per_cu=262144)


# -- Hardware constants per target ------------------------------------------

_TARGET_PARAMS = {
    "gfx940": {
        "lds_per_cu": 65536,
        "vgprs_per_simd": 512,
        "agprs_per_simd": 512,
        "num_simds": 4,
        "max_vgprs_per_wave": 256,
        "max_agprs_per_wave": 256,
        "wavefront_size": 64,
    },
    "gfx942": {
        "lds_per_cu": 65536,
        "vgprs_per_simd": 512,
        "agprs_per_simd": 512,
        "num_simds": 4,
        "max_vgprs_per_wave": 256,
        "max_agprs_per_wave": 256,
        "wavefront_size": 64,
    },
    "gfx950": {
        "lds_per_cu": 262144,
        "vgprs_per_simd": 512,
        "agprs_per_simd": 512,
        "num_simds": 4,
        "max_vgprs_per_wave": 256,
        "max_agprs_per_wave": 256,
        "wavefront_size": 64,
    },
}


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
    params = _TARGET_PARAMS.get(mcpu, _TARGET_PARAMS["gfx942"])
    wf = params["wavefront_size"]
    num_waves = (num_threads + wf - 1) // wf
    total_waves = num_waves * num_wg_per_cu
    waves_per_simd = (total_waves + params["num_simds"] - 1) // params["num_simds"]
    max_vgprs = min(
        params["max_vgprs_per_wave"],
        params["vgprs_per_simd"] // waves_per_simd,
    )
    max_agprs = min(
        params["max_agprs_per_wave"],
        params["agprs_per_simd"] // waves_per_simd,
    )
    lds_per_wg = params["lds_per_cu"] // num_wg_per_cu
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
