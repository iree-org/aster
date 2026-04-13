"""Kernel resource metadata parsing and register budget utilities.

Parses the .amdgpu_metadata YAML section emitted by the ASTER compiler
and exposes register/LDS occupancy helpers that are independent of
MLIR/LLVM.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aster.core.target import Target


@dataclass
class KernelResources:
    """Resource usage extracted from AMDGPU assembly metadata.

    This is the ground truth for what the hardware will actually use, as
    emitted by the ASTER compiler in .amdgpu_metadata.
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
        self,
        num_threads: int,
        *,
        mcpu: str,
        num_wg_per_cu: int = 1,
    ) -> List[str]:
        """Return the list of occupancy violations for ``mcpu``.

        ``mcpu`` is required (keyword-only) -- callers must pass the
        config's target (e.g. ``cfg.mcpu``).
        """
        target = Target.from_mcpu(mcpu)
        num_waves = (num_threads + target.wavefront_size - 1) // target.wavefront_size
        total_waves = num_waves * num_wg_per_cu
        waves_per_simd = (total_waves + target.num_simds - 1) // target.num_simds
        violations = []

        # Per-wave register limits (ISA manual: "A wave may have up to 512
        # total VGPRs, 256 of each type").
        if self.vgpr_count > target.max_vgprs:
            violations.append(f"vgpr per wave {self.vgpr_count} > {target.max_vgprs}")
        if target.max_agprs > 0 and self.agpr_count > target.max_agprs:
            violations.append(f"agpr per wave {self.agpr_count} > {target.max_agprs}")
        combined = self.vgpr_count + self.agpr_count
        combined_max = target.max_vgprs + target.max_agprs
        if combined > combined_max:
            violations.append(
                f"(vgpr+agpr) per wave {self.vgpr_count}+{self.agpr_count}"
                f"={combined} > {combined_max}"
            )
        # Per-CU register file constraint. The hardware CP rejects the dispatch
        # (EC code 22, HSA_STATUS_ERROR_INVALID_ISA) when total register lanes
        # exceed regsPerMultiprocessor (131072 on gfx942).
        # Total lanes = align(regs_per_wave, granule) * num_waves * wavefront_size.
        # Source: clr/rocclr/device/rocm/rocdevice.cpp:1604.
        if target.unified_reg_file and waves_per_simd > 1:
            g = target.vgpr_alloc_granule
            aligned = (combined + g - 1) // g * g
            wf = target.wavefront_size
            lanes_needed = aligned * total_waves * wf
            lanes_available = target.vgprs_per_simd * target.num_simds * wf
            if lanes_needed > lanes_available:
                violations.append(
                    f"CU reg file: align{g}({self.vgpr_count}+{self.agpr_count})"
                    f"={aligned} * {total_waves} waves"
                    f" ({num_waves} waves/WG * {num_wg_per_cu} WG/CU occupancy requested by user)"
                    f" * {wf} lanes = {lanes_needed}"
                    f" > regsPerMultiprocessor ({lanes_available})"
                )
        lds_budget = target.lds_per_cu // num_wg_per_cu
        if self.lds_bytes > lds_budget:
            violations.append(f"LDS per workgroup {self.lds_bytes} > {lds_budget}")
        return violations


def compute_register_budget(
    num_threads: int,
    mcpu: str = "gfx942",
    num_wg_per_cu: int = 1,
    agpr_hint: int = 0,
) -> Tuple[int, int, int]:
    """Compute per-wave VGPR/AGPR limits and per-WG LDS limit.

    On architectures with a unified register file (CDNA), VGPRs and AGPRs
    share a physical file per SIMD.  When multiple waves occupy a SIMD the
    per-wave combined budget must be reduced so that::

        align(combined, granule) * waves_per_simd <= vgprs_per_simd

    The returned (max_vgprs, max_agprs) are capped to the tightest of the
    per-wave ISA limit and the CU register-file constraint.  The split
    between VGPRs and AGPRs uses *agpr_hint* (expected AGPR usage) to
    maximise VGPRs after reserving what the accumulator tiles need.

    Args:
        num_threads: Number of threads per workgroup.
        mcpu: Target GPU (e.g. "gfx942", "gfx950").
        num_wg_per_cu: Number of workgroups sharing a CU (default 1).
        agpr_hint: Expected AGPR usage (e.g. from estimated_agprs).  Used
            to split the combined budget when there is a CU constraint.
            Defaults to 0 (fall back to 1/4 of combined budget for AGPRs).

    Returns:
        (max_vgprs, max_agprs, lds_per_wg) tuple.
    """
    try:
        target = Target.from_mcpu(mcpu)
    except ValueError:
        target = Target.from_mcpu("gfx942")

    max_vgprs = target.max_vgprs
    max_agprs = target.max_agprs

    # CU register-file constraint: when multiple waves share a SIMD the
    # per-wave combined budget shrinks.  Round down to the allocation
    # granule so the compiler never produces an HSACO that exceeds the CU
    # capacity.
    if target.unified_reg_file:
        wf = target.wavefront_size
        num_waves = (num_threads + wf - 1) // wf
        total_waves = num_waves * num_wg_per_cu
        waves_per_simd = (total_waves + target.num_simds - 1) // target.num_simds
        if waves_per_simd > 1:
            g = target.vgpr_alloc_granule
            # Max combined (aligned) registers per wave that fits in the SIMD.
            max_combined = (target.vgprs_per_simd // waves_per_simd // g) * g
            # Reserve AGPRs based on hint (or 1/4 of budget as fallback),
            # then give the rest to VGPRs which handle loads and addresses.
            agpr_reserve = max(agpr_hint, max_combined // 4)
            max_agprs = min(max_agprs, agpr_reserve)
            max_vgprs = min(max_vgprs, max_combined - max_agprs)

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

    meta_text is everything between the --- delimiters in
    .amdgpu_metadata.
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
