"""Kernel resource metadata parsing and register budget utilities.

Parses the .amdgpu_metadata YAML section emitted by the ASTER compiler and exposes
register/LDS occupancy helpers that are independent of MLIR/LLVM.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    def check_occupancy(
        self, num_threads: int, mcpu: str = "gfx942", num_wg_per_cu: int = 1
    ) -> List[str]:
        """Return a list of occupancy violations for the given target.

        Returns an empty list if the kernel can launch or the mcpu is unknown.
        """
        try:
            target = Target.from_mcpu(mcpu)
        except ValueError:
            return []
        num_waves = (num_threads + target.wavefront_size - 1) // target.wavefront_size
        total_waves = num_waves * num_wg_per_cu
        waves_per_simd = (total_waves + target.num_simds - 1) // target.num_simds
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
        if target.unified_reg_file:
            # CDNA3/CDNA4: VGPRs+AGPRs share a single physical file per SIMD.
            g = target.vgpr_alloc_granule
            aligned = (combined + g - 1) // g * g
            total_simd = aligned * waves_per_simd
            if total_simd > target.vgprs_per_simd:
                violations.append(
                    f"unified reg file per SIMD: aligned({self.vgpr_count}"
                    f"+{self.agpr_count}, {g})={aligned} * {waves_per_simd}"
                    f" waves = {total_simd} > {target.vgprs_per_simd}"
                )
        else:
            # Separate VGPR and AGPR files (RDNA).
            vgpr_simd = self.vgpr_count * waves_per_simd
            if vgpr_simd > target.vgprs_per_simd:
                violations.append(
                    f"vgpr per SIMD {self.vgpr_count}*{waves_per_simd}"
                    f"={vgpr_simd} > {target.vgprs_per_simd}"
                )
        lds_budget = target.lds_per_cu // num_wg_per_cu
        if self.lds_bytes > lds_budget:
            violations.append(f"LDS per workgroup {self.lds_bytes} > {lds_budget}")
        return violations


def compute_register_budget(
    num_threads: int,
    mcpu: str = "gfx942",
    num_wg_per_cu: int = 1,
) -> Tuple[int, int, int]:
    """Compute per-wave VGPR/AGPR limits and per-WG LDS limit from occupancy target.

    On CDNA3/CDNA4 (unified_reg_file=True), VGPRs and AGPRs share a single
    physical file per SIMD. The returned max_vgprs reflects the unified budget:
    vgpr + agpr must not exceed max_vgprs.

    Args:
        num_threads: Number of threads per workgroup.
        mcpu: Target GPU (e.g. "gfx942", "gfx950"). If the mcpu is not
            recognised, the function silently falls back to gfx942 so that
            callers can always obtain a budget estimate. Use
            :meth:`KernelResources.check_occupancy` if you need a strict
            per-architecture check (it returns an empty list for unknown mcpu
            rather than substituting a default).
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
    if target.unified_reg_file:
        # Unified file: total (vgpr+agpr) per wave capped by file / waves.
        max_total = target.vgprs_per_simd // waves_per_simd
        max_vgprs = min(target.max_vgprs, max_total)
        max_agprs = min(target.max_agprs, max_total)
    else:
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
        # Re-add the leading "." that was consumed by the split.
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
