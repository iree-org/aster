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
    """Linalg-style GEMM specification.

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
            self.mfma_tile_bytes_a = self.mfma_m * self.mfma_k * self.elt_bytes_a
        if self.mfma_tile_bytes_b is None:
            self.mfma_tile_bytes_b = self.mfma_n * self.mfma_k * self.elt_bytes_b
        if self.mfma_tile_bytes_c is None:
            self.mfma_tile_bytes_c = self.mfma_m * self.mfma_n * self.elt_bytes_c

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

    # --- MFMA shape accessors ---

    @property
    def mfma_m(self) -> int:
        return self.mfma_shape[0]

    @property
    def mfma_n(self) -> int:
        return self.mfma_shape[1]

    @property
    def mfma_k(self) -> int:
        return self.mfma_shape[2]

    # --- GEMM problem sizes (derived from layouts) ---

    def _layout_sizes(self, layout: Layout) -> tuple[int, ...]:
        s = layout.sizes
        return s if isinstance(s, tuple) else (s,)

    def _layout_strides(self, layout: Layout) -> tuple[int, ...]:
        s = layout.strides
        return s if isinstance(s, tuple) else (s,)

    def _dim_from_operand(self, operand: str, dim: int) -> int:
        dims, layout = self._operand_info(operand)
        return self._layout_sizes(layout)[dims.index(dim)]

    @property
    def GEMM_SIZE_M(self) -> int:
        return self._dim_from_operand("a", DIM_M)

    @property
    def GEMM_SIZE_N(self) -> int:
        return self._dim_from_operand("b", DIM_N)

    @property
    def GEMM_SIZE_K(self) -> int:
        return self._dim_from_operand("a", DIM_K)

    # --- Operand queries ---

    def operand_dim_size(self, operand: str, dim: int) -> int:
        return self._dim_from_operand(operand, dim)

    def operand_stride(self, operand: str, dim: int) -> int:
        dims, layout = self._operand_info(operand)
        return self._layout_strides(layout)[dims.index(dim)]

    def _operand_info(self, operand: str) -> tuple[list[int], Layout]:
        if operand == "a":
            return self.logical_dims_a, self.layout_a
        elif operand == "b":
            return self.logical_dims_b, self.layout_b
        elif operand == "c":
            return self.logical_dims_c, self.layout_c
        raise ValueError(f"Unknown operand: {operand!r}")

    @property
    def stride_a(self) -> int:
        return self.operand_stride("a", self.logical_dims_a[0])

    @property
    def stride_b(self) -> int:
        return self.operand_stride("b", self.logical_dims_b[0])

    @property
    def stride_c(self) -> int:
        return self.operand_stride("c", self.logical_dims_c[0])

    @property
    def total_flops(self) -> int:
        return 2 * self.GEMM_SIZE_M * self.GEMM_SIZE_N * self.GEMM_SIZE_K


# -----------------------------------------------------------------------
# GemmMappingSpec -- transform-dialect style mapping for GemmSpec
# -----------------------------------------------------------------------


@dataclass
class GemmMappingSpec:
    """Transform-dialect style mapping specification for a Linalg-like GemmSpec.

    Describes how to distribute a GEMM across the GPU hierarchy (workgroups,
    waves, tiles) and how to schedule the software pipeline. Orthogonal to
    the GemmSpec problem definition -- the same GemmSpec can be mapped in
    many different ways.

    Analogous to transform.structured.tile_using_forall + pipeline schedule
    annotations: the tiling sizes, distribution strategy, and memory path
    choices are all mapping decisions, not problem properties.
    """

    # --- Distribution ---

    # Workgroup grid: the problem is tiled into m_wg x n_wg workgroups.
    m_wg: int
    n_wg: int

    # Wave grid within a workgroup.
    m_waves: int
    n_waves: int

    # Tiles per wave per K-loop iteration.
    m_tiles_per_wave: int
    n_tiles_per_wave: int

    # K-tiles per K-loop iteration (number of transfer tiles along K).
    k_tiles: int

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
    def m_tiles_wg(self) -> int:
        return self.m_tiles_per_wave * self.m_waves

    @property
    def n_tiles_wg(self) -> int:
        return self.n_tiles_per_wave * self.n_waves

    # --- Grid sizes ---

    @property
    def num_workgroups(self) -> int:
        return self.m_wg * self.n_wg

    @property
    def num_waves(self) -> int:
        return self.m_waves * self.n_waves

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

    def _coop_2d_split(self, spatial_tiles: int) -> tuple[int, int, int, int]:
        waves_s = min(spatial_tiles, self.num_waves)
        waves_k = max(1, self.num_waves // waves_s)
        coop_s = -(-spatial_tiles // waves_s)
        coop_k = -(-self.k_tiles // waves_k)
        return waves_s, waves_k, coop_s, coop_k

    @property
    def coop_a_split(self) -> tuple[int, int, int, int]:
        return self._coop_2d_split(self.m_tiles_wg)

    @property
    def coop_b_split(self) -> tuple[int, int, int, int]:
        return self._coop_2d_split(self.n_tiles_wg)

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
        return self.m_tiles_per_wave * self.n_tiles_per_wave * 4

    def estimated_vgprs(self) -> int:
        a_load = self.coop_a_mk_count * self.a_stages * 4
        a_lds_read = self.m_tiles_per_wave * self.k_tiles * 4

        if self.direct_b:
            b_load = self.n_tiles_per_wave * self.k_tiles * self.b_stages * 4
            b_split = self.n_tiles_per_wave * self.k_tiles * 4
            overhead = 30
        else:
            b_load = self.coop_b_nk_count * self.a_stages * 4
            b_split = self.n_tiles_per_wave * self.k_tiles * 4
            overhead = 10

        return a_load + a_lds_read + b_load + b_split + overhead

    def lds_bytes(self, tile_bytes: int = 1024) -> int:
        a_lds = self.a_stages * self.m_tiles_wg * self.k_tiles * tile_bytes
        b_lds = (
            0
            if self.direct_b
            else self.b_stages * self.n_tiles_wg * self.k_tiles * tile_bytes
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
        GEMM_SIZE_M == m_wg * m_tiles_wg * mfma_m
        GEMM_SIZE_N == n_wg * n_tiles_wg * mfma_n

    All attributes delegate to spec or mapping via __getattr__.
    String-valued properties (load_type, b_path) return enum .value
    for backward compatibility with serde and MLIR template lookups.
    """

    def __init__(self, spec: GemmSpec, mapping: GemmMappingSpec):
        self.spec = spec
        self.mapping = mapping
        self._check_weak_scale()

    def _check_weak_scale(self) -> None:
        expected_m = self.mapping.m_wg * self.mapping.m_tiles_wg * self.spec.mfma_m
        expected_n = self.mapping.n_wg * self.mapping.n_tiles_wg * self.spec.mfma_n
        assert self.spec.GEMM_SIZE_M == expected_m, (
            f"GEMM_SIZE_M={self.spec.GEMM_SIZE_M} != "
            f"m_wg({self.mapping.m_wg}) * m_tiles_wg({self.mapping.m_tiles_wg}) "
            f"* mfma_m({self.spec.mfma_m}) = {expected_m}"
        )
        assert self.spec.GEMM_SIZE_N == expected_n, (
            f"GEMM_SIZE_N={self.spec.GEMM_SIZE_N} != "
            f"n_wg({self.mapping.n_wg}) * n_tiles_wg({self.mapping.n_tiles_wg}) "
            f"* mfma_n({self.spec.mfma_n}) = {expected_n}"
        )

    # --- Convenience accessors (backward compat) ---

    @property
    def k(self) -> int:
        return self.spec.GEMM_SIZE_K

    @property
    def m_dim(self) -> int:
        return self.spec.GEMM_SIZE_M

    @property
    def n_dim(self) -> int:
        return self.spec.GEMM_SIZE_N

    @property
    def load_type(self) -> str:
        return self.mapping.load_type.value

    @property
    def b_path(self) -> str:
        return self.mapping.operand_path.value

    @property
    def m_tiles(self) -> int:
        return self.mapping.m_tiles_per_wave

    @property
    def n_tiles(self) -> int:
        return self.mapping.n_tiles_per_wave

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
        return self.k // (self.mapping.k_tiles * 32)

    # --- Label serde ---

    _LABEL_RE = None

    @classmethod
    def _label_pattern(cls):
        if cls._LABEL_RE is None:
            import re

            cls._LABEL_RE = re.compile(
                r"^m(\d+)xn(\d+)xk(\d+)"
                r"_wg(\d+)x(\d+)_w(\d+)x(\d+)"
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
            m_wg,
            n_wg,
            m_waves,
            n_waves,
            m_tiles_wg,
            n_tiles_wg,
            k_tiles,
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
        mw, nw = int(m_waves), int(n_waves)
        mtw, ntw = int(m_tiles_wg), int(n_tiles_wg)

        spec = GemmSpec.from_sizes(int(_m_dim), int(_n_dim), int(k))
        mapping = GemmMappingSpec(
            m_wg=int(m_wg),
            n_wg=int(n_wg),
            m_waves=mw,
            n_waves=nw,
            m_tiles_per_wave=mtw // mw,
            n_tiles_per_wave=ntw // nw,
            k_tiles=int(k_tiles),
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
        tile_str = f"_twg{self.mapping.m_tiles_wg}x{self.mapping.n_tiles_wg}x{self.mapping.k_tiles}"
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
        return (
            f"m{self.m_dim}xn{self.n_dim}xk{self.k}"
            f"_wg{self.mapping.m_wg}x{self.mapping.n_wg}"
            f"_w{self.mapping.m_waves}x{self.mapping.n_waves}"
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
