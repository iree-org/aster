"""GPU target definitions."""

import enum
from dataclasses import dataclass
from typing import Dict, Optional


class GpuArch(enum.Enum):
    """Supported AMDGPU architectures."""

    GFX940 = "gfx940"
    GFX942 = "gfx942"
    GFX950 = "gfx950"
    GFX1201 = "gfx1201"


@dataclass(frozen=True)
class _ArchParams:
    """Hardware constants for an AMDGPU architecture."""

    wavefront_size: int
    lds_per_cu: int  # Bytes of LDS per compute unit.
    max_vgprs: int  # Max VGPRs per thread.
    max_agprs: int  # Max AGPRs per thread (0 on architectures without matrix cores).
    max_sgprs: int  # Max SGPRs per wavefront.
    num_simds: int  # Number of SIMD units per compute unit.
    vgprs_per_simd: int  # Total VGPR slots per SIMD unit (limits waves * vgprs).
    agprs_per_simd: int  # Total AGPR slots per SIMD unit (0 on RDNA).
    vgpr_alloc_granule: int  # VGPR allocation granularity (registers per block).
    unified_reg_file: bool  # True if VGPRs+AGPRs share one physical file per SIMD.


# Hardware constants sourced from the AMD ISA reference manuals and verified
# against LLVM's AMDGPUBaseInfo.cpp (getAddressableNumSGPRs / getTotalNumVGPRs).
# These are compile-time fallbacks for cross-compilation (macOS -> Linux).
# At runtime on GPU, use Target.from_device() to get the actual values from HIP
# (see aster.core.device.query_device).
_ARCH_PARAMS: Dict[GpuArch, _ArchParams] = {
    # CDNA3: MI300-series (GFX9-class).
    # SGPRs: 102 addressable per wavefront (GFX8-9 limit).
    # VGPRs+AGPRs: single 512-entry unified physical file per SIMD (ISA 3.6.4).
    # Allocation granularity: 8 registers (FeatureGFX90AInsts).
    GpuArch.GFX940: _ArchParams(
        wavefront_size=64,
        lds_per_cu=64 * 1024,
        max_vgprs=256,
        max_agprs=256,
        max_sgprs=102,
        num_simds=4,
        vgprs_per_simd=512,
        agprs_per_simd=512,
        vgpr_alloc_granule=8,
        unified_reg_file=True,
    ),
    # GFX940 and GFX942 are different MI300-series SKUs but share identical
    # register-file and LDS hardware constants.
    GpuArch.GFX942: _ArchParams(
        wavefront_size=64,
        lds_per_cu=64 * 1024,
        max_vgprs=256,
        max_agprs=256,
        max_sgprs=102,
        num_simds=4,
        vgprs_per_simd=512,
        agprs_per_simd=512,
        vgpr_alloc_granule=8,
        unified_reg_file=True,
    ),
    # CDNA4: gfx950 (GFX9-class, same register file as CDNA3, larger LDS split
    # into 64 banks instead of 32). LDS is 160 KB per CU.
    GpuArch.GFX950: _ArchParams(
        wavefront_size=64,
        lds_per_cu=160 * 1024,
        max_vgprs=256,
        max_agprs=256,
        max_sgprs=102,
        num_simds=4,
        vgprs_per_simd=512,
        agprs_per_simd=512,
        vgpr_alloc_granule=8,
        unified_reg_file=True,
    ),
    # RDNA4: gfx1201 (GFX10+-class, wave32, no AGPRs).
    # SGPRs: 106 addressable per wavefront (GFX10+ limit).
    # VGPRs: 1024 addressable per wavefront (wave32, Feature1024AddressableVGPRs).
    GpuArch.GFX1201: _ArchParams(
        wavefront_size=32,
        lds_per_cu=128 * 1024,
        max_vgprs=1024,
        max_agprs=0,
        max_sgprs=106,
        num_simds=4,
        vgprs_per_simd=1024,
        agprs_per_simd=0,
        vgpr_alloc_granule=8,
        unified_reg_file=False,
    ),
}


@dataclass
class Target:
    """Compilation target: GPU architecture and wavefront size."""

    arch: GpuArch
    wavefront_size: int

    @property
    def mcpu(self) -> str:
        """Return the target architecture string (e.g. 'gfx942')."""
        return self.arch.value

    @property
    def lds_per_cu(self) -> int:
        """Bytes of LDS available per compute unit."""
        return _ARCH_PARAMS[self.arch].lds_per_cu

    @property
    def max_vgprs(self) -> int:
        """Max vector GPRs per thread."""
        return _ARCH_PARAMS[self.arch].max_vgprs

    @property
    def max_agprs(self) -> int:
        """Max accumulation GPRs per thread (0 on architectures without matrix cores)."""
        return _ARCH_PARAMS[self.arch].max_agprs

    @property
    def max_sgprs(self) -> int:
        """Max scalar GPRs per wavefront."""
        return _ARCH_PARAMS[self.arch].max_sgprs

    @property
    def num_simds(self) -> int:
        """Number of SIMD units per compute unit."""
        return _ARCH_PARAMS[self.arch].num_simds

    @property
    def vgprs_per_simd(self) -> int:
        """Total VGPR slots per SIMD unit (limits waves * vgprs_per_wave)."""
        return _ARCH_PARAMS[self.arch].vgprs_per_simd

    @property
    def agprs_per_simd(self) -> int:
        """Total AGPR slots per SIMD unit (0 on RDNA architectures)."""
        return _ARCH_PARAMS[self.arch].agprs_per_simd

    @property
    def vgpr_alloc_granule(self) -> int:
        """VGPR allocation granularity (registers per block)."""
        return _ARCH_PARAMS[self.arch].vgpr_alloc_granule

    @property
    def unified_reg_file(self) -> bool:
        """True if VGPRs and AGPRs share one physical file per SIMD."""
        return _ARCH_PARAMS[self.arch].unified_reg_file

    @classmethod
    def from_mcpu(cls, mcpu: str, wavefront_size: Optional[int] = None) -> "Target":
        """Construct a Target from an mcpu string.

        Args:
            mcpu: Architecture string (e.g. 'gfx942').
            wavefront_size: Wavefront size. Defaults to the architecture default.
        """
        arch = GpuArch(mcpu)
        wf = (
            wavefront_size
            if wavefront_size is not None
            else _ARCH_PARAMS[arch].wavefront_size
        )
        return cls(arch=arch, wavefront_size=wf)

    @classmethod
    def from_device(cls, device_id: int = 0) -> "Target":
        """Construct a Target from the actual GPU via HIP device properties.

        Queries the hardware at runtime and overrides the hardcoded
        _ArchParams with the real values. Raises RuntimeError if HIP is
        unavailable.
        """
        from aster.core.device import query_device

        dp = query_device(device_id)
        arch = GpuArch(dp.gcn_arch_name)
        # Override the static table with runtime values.
        _ARCH_PARAMS[arch] = _ArchParams(
            wavefront_size=dp.warp_size,
            lds_per_cu=dp.lds_per_cu,
            max_vgprs=dp.vgprs_per_simd,  # max per wave = file size (at occupancy 1)
            max_agprs=dp.vgprs_per_simd,  # symmetric on CDNA
            max_sgprs=_ARCH_PARAMS.get(arch, _ARCH_PARAMS[GpuArch.GFX942]).max_sgprs,
            num_simds=dp.num_simds,
            vgprs_per_simd=dp.vgprs_per_simd,
            agprs_per_simd=dp.vgprs_per_simd,
            vgpr_alloc_granule=dp.vgpr_alloc_granule,
            unified_reg_file=not dp.gcn_arch_name.startswith(
                "gfx12"
            ),  # CDNA = unified, RDNA4 = separate
        )
        return cls(arch=arch, wavefront_size=dp.warp_size)

    @classmethod
    def gfx942(cls, wavefront_size: int = 64) -> "Target":
        return cls(GpuArch.GFX942, wavefront_size)

    @classmethod
    def gfx950(cls, wavefront_size: int = 64) -> "Target":
        return cls(GpuArch.GFX950, wavefront_size)

    @classmethod
    def gfx1201(cls, wavefront_size: int = 32) -> "Target":
        return cls(GpuArch.GFX1201, wavefront_size)
