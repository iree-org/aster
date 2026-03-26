# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Query GPU hardware constants from HIP at runtime.

On machines with a GPU and HIP runtime, this module queries the actual hardware
properties via hipGetDeviceProperties and derives the constants that target.py
hardcodes. This is the source of truth -- the hardcoded values in target.py are
compile-time fallbacks for cross-compilation (macOS -> Linux).

Usage:
    from aster.core.device import query_device, DeviceProps

    props = query_device(0)
    print(props.vgprs_per_simd)   # 512 on gfx942
    print(props.lds_per_cu)       # 65536 on gfx942
    print(props.max_waves_per_cu) # 32 on gfx942

Source: clr/rocclr/device/rocm/rocdevice.cpp:1593-1610
  availableRegistersPerCU_ = vgprsPerSimd_ * simdPerCU_ * wavefrontWidth_
  Exposed as hipDeviceProp_t.regsPerMultiprocessor
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceProps:
    """Hardware constants queried from HIP at runtime."""

    name: str
    gcn_arch_name: str

    # Wavefront (warp) size: 64 on CDNA, 32 on RDNA.
    warp_size: int

    # LDS per compute unit (bytes).
    lds_per_cu: int

    # Register file: regsPerMultiprocessor = vgprsPerSimd * simdPerCU * warpSize.
    regs_per_multiprocessor: int

    # CU count.
    multiprocessor_count: int

    # Max threads per workgroup.
    max_threads_per_block: int

    # Max threads per CU (= max waves per CU * warpSize).
    max_threads_per_multiprocessor: int

    # -- Derived constants --

    @property
    def num_simds(self) -> int:
        """Number of SIMDs per CU.

        Always 4 on current AMD GPUs.
        """
        return 4

    @property
    def vgprs_per_simd(self) -> int:
        """Architectural VGPRs per SIMD (512 on gfx942).

        Derived from regsPerMultiprocessor / warpSize / numSIMDs. regsPerMultiprocessor
        counts 32-bit register *lanes* across all SIMDs. Dividing by warpSize gives
        architectural registers per CU, then by numSIMDs gives per-SIMD.
        """
        return self.regs_per_multiprocessor // self.warp_size // self.num_simds

    @property
    def max_waves_per_simd(self) -> int:
        """Max waves per SIMD (8 on gfx942)."""
        return self.max_threads_per_multiprocessor // self.warp_size // self.num_simds

    @property
    def max_waves_per_cu(self) -> int:
        """Max waves per CU (32 on gfx942)."""
        return self.max_threads_per_multiprocessor // self.warp_size

    @property
    def vgpr_alloc_granule(self) -> int:
        """VGPR allocation granularity.

        8 on all current AMD GPUs.
        """
        return 8


_cache: dict = {}


def query_device(device_id: int = 0) -> DeviceProps:
    """Query hardware properties from HIP for the given device.

    Results are cached per device_id. Raises RuntimeError if HIP is not available or the
    device does not exist.
    """
    if device_id in _cache:
        return _cache[device_id]

    from aster._mlir_libs._runtime_module import hip_init, hip_get_device_props

    hip_init()
    raw = hip_get_device_props(device_id)
    props = DeviceProps(
        name=raw["name"],
        gcn_arch_name=raw["gcn_arch_name"],
        warp_size=raw["warp_size"],
        lds_per_cu=raw["lds_per_cu"],
        regs_per_multiprocessor=raw["regs_per_multiprocessor"],
        multiprocessor_count=raw["multiprocessor_count"],
        max_threads_per_block=raw["max_threads_per_block"],
        max_threads_per_multiprocessor=raw["max_threads_per_multiprocessor"],
    )
    _cache[device_id] = props
    return props


def try_query_device(device_id: int = 0) -> Optional[DeviceProps]:
    """Like query_device but returns None if HIP is unavailable."""
    try:
        return query_device(device_id)
    except Exception:
        return None
