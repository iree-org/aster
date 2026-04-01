# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""GEMM specification, GPU mapping, and weak-scale configuration.

Three orthogonal classes:

GemmSpec        -- the GEMM problem (Linalg-style: layouts, sizes, mfma shape).
                   Pure math, no GPU concepts.

GemmMappingSpec -- how to map the GEMM onto GPU hierarchy (WG grid, wave grid,
                   tiles per wave, pipeline strategy, memory paths).
                   Provides resource estimates (LDS, VGPRs, AGPRs).

WeakScaledMappedGemmInstance -- takes a GemmSpec + GemmMappingSpec, asserts weak-scale
                   consistency, provides label serde for sweep infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aster.layout import Layout


# -----------------------------------------------------------------------
# Enums for memory path choices
# -----------------------------------------------------------------------


class LoadType(Enum):
    FLAT = "flat"
    BUFFER = "buffer"


class OperandPath(Enum):
    LDS = "lds"  # both A and B through LDS
    DIRECT_B = "direct_b"  # B bypasses LDS (bpermute)
    DIRECT_AB = "direct_ab"  # both A and B bypass LDS


class Operand(Enum):
    A = "a"
    B = "b"
    C = "c"


# Shorthand constants for use at call sites.
A = Operand.A
B = Operand.B
C = Operand.C


# -----------------------------------------------------------------------
# Iteration dimension markers
# -----------------------------------------------------------------------

DIM_M = 0
DIM_N = 1
DIM_K = 2


# -----------------------------------------------------------------------
# GemmSpec -- the GEMM problem
# -----------------------------------------------------------------------


@dataclass
class GemmSpec:
    """Linalg-style GEMM specification, from a "generative" perspective.

    Defines C[dm, dn] += A[dm, dk] * B[dn, dk]. Layouts are the source of truth for
    problem dimensions. GEMM_SIZE_M/N/K are derived from layouts.
    """

    # Per-operand memory layouts (source of truth for sizes and strides).
    layout_a: Layout
    layout_b: Layout
    layout_c: Layout

    # Per-operand logical shapes: list of DIM_* indices mapping iteration
    # dims to operand axes. Permute for transposed variants.
    logical_dims_a: list[int] = field(default_factory=lambda: [DIM_M, DIM_K])
    logical_dims_b: list[int] = field(default_factory=lambda: [DIM_N, DIM_K])
    logical_dims_c: list[int] = field(default_factory=lambda: [DIM_M, DIM_N])

    # MFMA tile shape [mfma_m, mfma_n, mfma_k].
    mfma_shape: list[int] = field(default_factory=lambda: [16, 16, 16])

    # Per-operand element byte widths.
    elt_bytes_a: int = 2  # f16
    elt_bytes_b: int = 2  # f16
    elt_bytes_c: int = 4  # f32

    # Per-operand MFMA tile byte counts (for subbyte types, set explicitly).
    mfma_tile_bytes_a: int | None = None
    mfma_tile_bytes_b: int | None = None
    mfma_tile_bytes_c: int | None = None

    def __post_init__(self):
        if self.mfma_tile_bytes_a is None:
            self.mfma_tile_bytes_a = (
                self.mfma_shape[DIM_M] * self.mfma_shape[DIM_K] * self.elt_bytes_a
            )
        if self.mfma_tile_bytes_b is None:
            self.mfma_tile_bytes_b = (
                self.mfma_shape[DIM_K] * self.mfma_shape[DIM_N] * self.elt_bytes_b
            )
        if self.mfma_tile_bytes_c is None:
            self.mfma_tile_bytes_c = (
                self.mfma_shape[DIM_M] * self.mfma_shape[DIM_N] * self.elt_bytes_c
            )

    @classmethod
    def from_sizes(
        cls,
        M: int,
        N: int,
        K: int,
        *,
        elt_bytes_a: int = 2,
        elt_bytes_b: int = 2,
        elt_bytes_c: int = 4,
        logical_dims_a: list[int] | None = None,
        logical_dims_b: list[int] | None = None,
        logical_dims_c: list[int] | None = None,
        **kwargs,
    ) -> GemmSpec:
        """Build a GemmSpec with packed row-major layouts from problem sizes."""
        from aster.layout import Layout

        dims_a = logical_dims_a or [DIM_M, DIM_K]
        dims_b = logical_dims_b or [DIM_N, DIM_K]
        dims_c = logical_dims_c or [DIM_M, DIM_N]
        dim_sizes = {DIM_M: M, DIM_N: N, DIM_K: K}

        def _packed(dims, eb):
            sizes = tuple(dim_sizes[d] for d in dims)
            strides, s = [], eb
            for sz in reversed(sizes):
                strides.append(s)
                s *= sz
            strides.reverse()
            return Layout(sizes, tuple(strides))

        return cls(
            layout_a=_packed(dims_a, elt_bytes_a),
            layout_b=_packed(dims_b, elt_bytes_b),
            layout_c=_packed(dims_c, elt_bytes_c),
            logical_dims_a=dims_a,
            logical_dims_b=dims_b,
            logical_dims_c=dims_c,
            elt_bytes_a=elt_bytes_a,
            elt_bytes_b=elt_bytes_b,
            elt_bytes_c=elt_bytes_c,
            **kwargs,
        )

    # --- GEMM problem sizes (derived from layouts) ---

    def _layout_sizes(self, layout: Layout) -> tuple[int, ...]:
        s = layout.sizes
        return s if isinstance(s, tuple) else (s,)

    def _layout_strides(self, layout: Layout) -> tuple[int, ...]:
        s = layout.strides
        return s if isinstance(s, tuple) else (s,)

    def _dim_from_operand(self, operand: Operand, dim: int) -> int:
        dims, layout = self._operand_info(operand)
        return self._layout_sizes(layout)[dims.index(dim)]

    @property
    def gemm_size(self) -> list[int]:
        """[M, N, K] GEMM problem dimensions derived from layouts."""
        return [
            self._dim_from_operand(Operand.A, DIM_M),
            self._dim_from_operand(Operand.B, DIM_N),
            self._dim_from_operand(Operand.A, DIM_K),
        ]

    # --- Operand queries ---

    def operand_shape(self, operand: Operand) -> tuple[int, ...]:
        """Shape of an operand matrix following its logical_dims ordering."""
        dims, layout = self._operand_info(operand)
        return self._layout_sizes(layout)

    def operand_dim_size(self, operand: Operand, dim: int) -> int:
        return self._dim_from_operand(operand, dim)

    def operand_stride(self, operand: Operand, dim: int) -> int:
        dims, layout = self._operand_info(operand)
        return self._layout_strides(layout)[dims.index(dim)]

    def _operand_info(self, operand: Operand) -> tuple[list[int], Layout]:
        if operand == Operand.A:
            return self.logical_dims_a, self.layout_a
        elif operand == Operand.B:
            return self.logical_dims_b, self.layout_b
        elif operand == Operand.C:
            return self.logical_dims_c, self.layout_c
        raise ValueError(f"Unknown operand: {operand!r}")

    @property
    def stride_a(self) -> int:
        return self.operand_stride(Operand.A, self.logical_dims_a[0])

    @property
    def stride_b(self) -> int:
        return self.operand_stride(Operand.B, self.logical_dims_b[0])

    @property
    def stride_c(self) -> int:
        return self.operand_stride(Operand.C, self.logical_dims_c[0])

    @property
    def total_flops(self) -> int:
        gs = self.gemm_size
        return 2 * gs[DIM_M] * gs[DIM_N] * gs[DIM_K]


# -----------------------------------------------------------------------
# GemmMappingSpec -- transform-dialect style mapping for GemmSpec
# -----------------------------------------------------------------------


@dataclass
class GemmMappingSpec:
    """Transform-dialect style mapping specification for a Linalg-like GemmSpec.

    Analogous to transform.structured.tile_using_forall + pipeline schedule
    annotations: the tiling sizes, distribution strategy, and memory path
    choices are all mapping decisions, not problem properties.
    """

    # --- Distribution ---

    # Workgroup grid [M, N, K]: number of workgroups along each dimension.
    # K is always 1 (no K-splitting across workgroups).
    num_workgroups_per_kernel: list[int]

    # Wave grid [M, N, K] within a workgroup. K is always 1.
    num_waves_per_workgroup: list[int]

    # Tiles per wave [M, N, K] per K-loop iteration.
    # K dimension is the number of transfer tiles along K.
    num_tiles_per_wave: list[int]

    # --- Pipeline schedule ---

    # Pipeline strategy index. Determines the stage assignment for each
    # operation (A_LOAD, B_LOAD, A_LDS_WRITE, ...). a_stages and b_stages
    # are derived from this via pipeline_strategy_stages().
    pipeline_strategy: int = 1

    # --- Memory path choices ---

    load_type: LoadType = LoadType.FLAT
    operand_path: OperandPath = OperandPath.LDS

    # --- Scheduling knobs ---

    lcm_unroll: bool = True  # LCM-based kernel loop unrolling
    unroll_factor_multiplier: int = 1  # extra unroll on top of LCM
    epilogue_peeling: bool = True  # fully unroll cleanup loop
    ll_sched: bool = False  # low-latency scheduling
    hoist_wait: bool = False  # hoist iter-arg waits

    # --- Occupancy ---

    num_wg_per_cu: int = 1
    wave_size: int = 64

    # --- Pipeline stages (derived from pipeline_strategy) ---

    @property
    def a_stages(self) -> int:
        a, _ = self._pipeline_stages()
        return a

    @property
    def b_stages(self) -> int:
        _, b = self._pipeline_stages()
        return b

    def _pipeline_stages(self) -> tuple[int, int]:
        from kittens_helpers import pipeline_strategy_stages

        return pipeline_strategy_stages(self.pipeline_strategy)

    @property
    def effective_b_stages(self) -> int:
        return self.b_stages

    @property
    def pipeline_depth(self) -> int:
        return max(self.a_stages, self.b_stages)

    # --- Derived tile counts ---

    @property
    def num_tiles_per_workgroup(self) -> list[int]:
        """[M, N, K] tile counts per workgroup."""
        wpw = self.num_waves_per_workgroup
        return [
            self.num_tiles_per_wave[DIM_M] * wpw[DIM_M],
            self.num_tiles_per_wave[DIM_N] * wpw[DIM_N],
            self.num_tiles_per_wave[DIM_K] * wpw[DIM_K],
        ]

    # --- Grid sizes ---

    @property
    def num_workgroups(self) -> int:
        return (
            self.num_workgroups_per_kernel[DIM_M]
            * self.num_workgroups_per_kernel[DIM_N]
            * self.num_workgroups_per_kernel[DIM_K]
        )

    @property
    def num_waves(self) -> int:
        wpw = self.num_waves_per_workgroup
        return wpw[DIM_M] * wpw[DIM_N] * wpw[DIM_K]

    @property
    def num_threads(self) -> int:
        return self.num_waves * self.wave_size

    # --- Memory path queries ---

    @property
    def direct_b(self) -> bool:
        return self.operand_path in (OperandPath.DIRECT_B, OperandPath.DIRECT_AB)

    @property
    def direct_a(self) -> bool:
        return self.operand_path == OperandPath.DIRECT_AB

    @property
    def use_buffer(self) -> bool:
        return self.load_type == LoadType.BUFFER

    # --- Cooperative loading splits ---

    def _coop_2d_split(self, num_tiles: int) -> tuple[int, int, int, int]:
        waves_s = min(num_tiles, self.num_waves)
        waves_k = max(1, self.num_waves // waves_s)
        coop_s = -(-num_tiles // waves_s)
        coop_k = -(-self.num_tiles_per_wave[DIM_K] // waves_k)
        return waves_s, waves_k, coop_s, coop_k

    @property
    def coop_a_split(self) -> tuple[int, int, int, int]:
        return self._coop_2d_split(self.num_tiles_per_workgroup[DIM_M])

    @property
    def coop_b_split(self) -> tuple[int, int, int, int]:
        return self._coop_2d_split(self.num_tiles_per_workgroup[DIM_N])

    @property
    def coop_a_mk_count(self) -> int:
        _, _, cm, ck = self.coop_a_split
        return cm * ck

    @property
    def coop_b_nk_count(self) -> int:
        _, _, cn, ck = self.coop_b_split
        return cn * ck

    # --- Resource estimates ---

    def estimated_agprs(self) -> int:
        return self.num_tiles_per_wave[DIM_M] * self.num_tiles_per_wave[DIM_N] * 4

    def estimated_vgprs(self) -> int:
        mt, nt, kt = self.num_tiles_per_wave
        a_load = self.coop_a_mk_count * self.a_stages * 4
        a_lds_read = mt * kt * 4

        if self.direct_b:
            b_load = kt * nt * self.b_stages * 4
            b_split = kt * nt * 4
            overhead = 30
        else:
            b_load = self.coop_b_nk_count * self.a_stages * 4
            b_split = kt * nt * 4
            overhead = 10

        return a_load + a_lds_read + b_load + b_split + overhead

    def lds_bytes(self, tile_bytes: int = 1024) -> int:
        kt = self.num_tiles_per_wave[DIM_K]
        a_lds = self.a_stages * self.num_tiles_per_workgroup[DIM_M] * kt * tile_bytes
        b_lds = (
            0
            if self.direct_b
            else self.b_stages * kt * self.num_tiles_per_workgroup[DIM_N] * tile_bytes
        )
        return a_lds + b_lds

    def simd_occupancy(self) -> int:
        import math

        return self.num_wg_per_cu * math.ceil(self.num_waves / 4)


# -----------------------------------------------------------------------
# WeakScaledMappedGemmInstance -- a validated (spec, mapping) pair
# -----------------------------------------------------------------------


class WeakScaledMappedGemmInstance:
    """A weak-scaled GEMM instance: a GemmSpec + GemmMappingSpec pair.

    Validates that the GEMM problem dimensions are consistent with the
    weak-scale distribution:
        GEMM_SIZE_M == num_workgroups_per_kernel[M] * num_tiles_per_workgroup[M] * mfma_m
        GEMM_SIZE_N == num_workgroups_per_kernel[N] * num_tiles_per_workgroup[N] * mfma_n

    All attributes delegate to spec or mapping via __getattr__.
    String-valued properties (load_type, b_path) return enum .value
    for backward compatibility with serde and MLIR template lookups.
    """

    def __init__(self, spec: GemmSpec, mapping: GemmMappingSpec):
        self.spec = spec
        self.mapping = mapping
        self._check_weak_scale()

    def _check_weak_scale(self) -> None:
        wg = self.mapping.num_workgroups_per_kernel
        twg = self.mapping.num_tiles_per_workgroup
        gs = self.spec.gemm_size
        mfma = self.spec.mfma_shape
        for dim, name in [(DIM_M, "M"), (DIM_N, "N")]:
            expected = wg[dim] * twg[dim] * mfma[dim]
            assert gs[dim] == expected, (
                f"gemm_size[{name}]={gs[dim]} != "
                f"num_workgroups_per_kernel[{name}]({wg[dim]}) * "
                f"num_tiles_per_wg[{name}]({twg[dim]}) * "
                f"mfma_shape[{name}]({mfma[dim]}) = {expected}"
            )

    # --- Convenience accessors ---

    @property
    def gemm_size(self) -> list[int]:
        return self.spec.gemm_size

    @property
    def load_type(self) -> str:
        return self.mapping.load_type.value

    @property
    def b_path(self) -> str:
        return self.mapping.operand_path.value

    @property
    def estimated_agprs(self) -> int:
        return self.mapping.estimated_agprs()

    @property
    def estimated_vgprs(self) -> int:
        return self.mapping.estimated_vgprs()

    @property
    def lds_bytes(self) -> int:
        return self.mapping.lds_bytes()

    @property
    def simd_occupancy(self) -> int:
        return self.mapping.simd_occupancy()

    _KERNEL_NAMES = {
        OperandPath.LDS: "gemm_f16_weak_scaled",
        OperandPath.DIRECT_B: "gemm_f16_direct_b",
        OperandPath.DIRECT_AB: "gemm_f16_direct_ab",
    }

    @property
    def kernel_name(self) -> str:
        return self._KERNEL_NAMES[self.mapping.operand_path]

    @property
    def k_scaling_factor(self) -> int:
        return self.gemm_size[DIM_K] // (self.mapping.num_tiles_per_wave[DIM_K] * 32)

    # --- Label serde ---

    _LABEL_RE = None

    @classmethod
    def _label_pattern(cls):
        if cls._LABEL_RE is None:
            import re

            cls._LABEL_RE = re.compile(
                r"^m(\d+)xn(\d+)xk(\d+)"
                r"_wg(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)x(\d+)"
                r"_twg(\d+)x(\d+)x(\d+)_pipestrat(\d+)"
                r"(?:_wgcu(\d+))?"
                r"(_nolcm)?"
                r"(?:_um(\d+))?"
                r"(_nopeel)?"
                r"(_llsched)?"
                r"(_hoistwait)?"
                r"_(?:(direct_ab|direct_b)_)?(flat|buf)$"
            )
        return cls._LABEL_RE

    @classmethod
    def from_label(cls, label: str) -> WeakScaledMappedGemmInstance:
        """Parse a label string back into an instance."""
        m = cls._label_pattern().match(label)
        if not m:
            raise ValueError(f"Cannot parse label: {label}")
        (
            _m_dim,
            _n_dim,
            k,
            wg_m,
            wg_n,
            wg_k,
            waves_m,
            waves_n,
            waves_k,
            twg_m,
            twg_n,
            twg_k,
            pipestrat,
            wgcu,
            nolcm,
            um,
            nopeel,
            llsched,
            hoistwait,
            direct,
            lt,
        ) = m.groups()

        load_type = "buffer" if lt == "buf" else "flat"
        b_path = direct if direct else "lds"
        wg = [int(wg_m), int(wg_n), int(wg_k)]
        wpw = [int(waves_m), int(waves_n), int(waves_k)]
        tiles_wg = [int(twg_m), int(twg_n), int(twg_k)]

        spec = GemmSpec.from_sizes(int(_m_dim), int(_n_dim), int(k))
        mapping = GemmMappingSpec(
            num_workgroups_per_kernel=wg,
            num_waves_per_workgroup=wpw,
            num_tiles_per_wave=[
                tiles_wg[DIM_M] // wpw[DIM_M],
                tiles_wg[DIM_N] // wpw[DIM_N],
                tiles_wg[DIM_K] // wpw[DIM_K],
            ],
            pipeline_strategy=int(pipestrat),
            load_type=LoadType(load_type),
            operand_path=OperandPath(b_path),
            num_wg_per_cu=int(wgcu) if wgcu else 1,
            lcm_unroll=nolcm is None,
            unroll_factor_multiplier=int(um) if um else 1,
            epilogue_peeling=nopeel is None,
            ll_sched=llsched is not None,
            hoist_wait=hoistwait is not None,
        )
        cfg = cls(spec, mapping)
        assert cfg.label == label, f"Round-trip failed: {cfg.label!r} != {label!r}"
        return cfg

    @property
    def label(self) -> str:
        wg = self.mapping.num_workgroups_per_kernel
        twg = self.mapping.num_tiles_per_workgroup
        tile_str = f"_twg{twg[DIM_M]}x{twg[DIM_N]}x{twg[DIM_K]}"
        lcm = "" if self.mapping.lcm_unroll else "_nolcm"
        um = (
            f"_um{self.mapping.unroll_factor_multiplier}"
            if self.mapping.unroll_factor_multiplier > 1
            else ""
        )
        peel = "" if self.mapping.epilogue_peeling else "_nopeel"
        llsched = "_llsched" if self.mapping.ll_sched else ""
        hoistwait = "_hoistwait" if self.mapping.hoist_wait else ""
        wgcu = (
            f"_wgcu{self.mapping.num_wg_per_cu}"
            if self.mapping.num_wg_per_cu != 1
            else ""
        )
        lt = "buf" if self.load_type == "buffer" else "flat"
        suffix = f"_{self.b_path}_{lt}" if self.b_path != "lds" else f"_{lt}"
        gs = self.gemm_size
        return (
            f"m{gs[DIM_M]}xn{gs[DIM_N]}xk{gs[DIM_K]}"
            f"_wg{wg[DIM_M]}x{wg[DIM_N]}x{wg[DIM_K]}"
            f"_w{self.mapping.num_waves_per_workgroup[DIM_M]}x{self.mapping.num_waves_per_workgroup[DIM_N]}x{self.mapping.num_waves_per_workgroup[DIM_K]}"
            f"{tile_str}_pipestrat{self.mapping.pipeline_strategy}"
            f"{wgcu}{lcm}{um}{peel}{llsched}{hoistwait}{suffix}"
        )

    # --- Fallback delegation ---

    def __getattr__(self, name: str):
        # Guard against infinite recursion during pickle unpickling:
        # self.spec/self.mapping may not exist yet.
        if name in ("spec", "mapping"):
            raise AttributeError(name)
        try:
            return getattr(object.__getattribute__(self, "spec"), name)
        except AttributeError:
            pass
        try:
            return getattr(object.__getattribute__(self, "mapping"), name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from None
