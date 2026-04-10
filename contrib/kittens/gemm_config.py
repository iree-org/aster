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
    """Linalg-style GEMM problem specification, from a "generative / bottom-up compositional" perspective.

    Defines C[dm, dn] += A[dm, dk] * B[dn, dk].

    Per-operand queries (operand_layout, operand_memory_layout,
    operand_logical_dims, operand_shape, operand_elt_bytes) provide
    uniform access to all operands.
    """

    # Per-operand memory layouts (source of truth for sizes and strides).
    layout_a: Layout
    layout_b: Layout
    layout_c: Layout

    # Per-operand logical shapes, permute for transposed variants.
    # TODO: Richer semantics via affine exprs.
    logical_dims_a: list[int] = field(default_factory=lambda: [DIM_M, DIM_K])
    logical_dims_b: list[int] = field(default_factory=lambda: [DIM_N, DIM_K])
    logical_dims_c: list[int] = field(default_factory=lambda: [DIM_M, DIM_N])

    # MFMA tile shape [mfma_m, mfma_n, mfma_k].
    mfma_shape: list[int] = field(default_factory=lambda: [16, 16, 16])

    # Per-operand element byte widths.
    elt_bytes_a: int = 2  # f16
    elt_bytes_b: int = 2  # f16
    elt_bytes_c: int = 4  # f32

    def __post_init__(self):
        self._operand_layouts = {
            Operand.A: OperandLayout(
                self.layout_a, self.logical_dims_a, self.elt_bytes_a, self.mfma_shape
            ),
            Operand.B: OperandLayout(
                self.layout_b, self.logical_dims_b, self.elt_bytes_b, self.mfma_shape
            ),
            Operand.C: OperandLayout(
                self.layout_c, self.logical_dims_c, self.elt_bytes_c, self.mfma_shape
            ),
        }

    def operand_layout(self, operand: Operand) -> OperandLayout:
        return self._operand_layouts[operand]

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

    # --- GEMM problem sizes (derived from per-operand layouts) ---

    def _dim_from_operand(self, operand: Operand, dim: int) -> int:
        ol = self.operand_layout(operand)
        return ol.shape[ol.logical_dims.index(dim)]

    @property
    def gemm_size(self) -> list[int]:
        """[M, N, K] GEMM problem dimensions derived from layouts."""
        return [
            self._dim_from_operand(Operand.A, DIM_M),
            self._dim_from_operand(Operand.B, DIM_N),
            self._dim_from_operand(Operand.A, DIM_K),
        ]

    # --- Operand queries (delegate to per-operand OperandLayout) ---

    def operand_shape(self, operand: Operand) -> tuple[int, ...]:
        """Shape of an operand matrix following its logical_dims ordering."""
        return self.operand_layout(operand).shape

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

    # Per-dimension tile multiplicity: elements per tile = mfma_shape * tile_mult.
    tile_mult: list[int] = field(default_factory=lambda: [1, 1, 2])

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
    set_mfma_priority: bool = True  # insert s_setprio around MFMA groups

    # --- LDS allocation strategy ---
    lds_at_write: bool = (
        False  # alloc at WRITE stage (fewer buffers, needs WAR barrier)
    )
    dealloc_at_read: bool = False  # dealloc at READ (test_102) vs COMPUTE (test_001)

    # --- Atomic transfer sizes (bytes per lane per memory operation) ---

    # Bytes per lane per memory operation.
    global_load_bytes: int = 16  # dwordx4 = 16 bytes per lane
    ds_write_bytes: int = 8  # ds_write_b64
    ds_read_bytes: int = 8  # ds_read_b64

    # --- LDS swizzle ---

    lds_swizzle_bits: int = 3
    lds_swizzle_base: int = 3
    lds_swizzle_shift: int = 3

    # --- Target ---

    mcpu: str = "gfx942"

    # --- Occupancy ---

    num_wg_per_cu: int = 1
    wave_size: int = 64

    def _pipeline_stage_dict(self) -> dict[str, int]:
        from kittens_helpers import PIPELINE_STRATEGIES

        return PIPELINE_STRATEGIES[self.pipeline_strategy]

    @property
    def isa(self) -> str:
        _MCPU_TO_ISA = {"gfx942": "cdna3", "gfx950": "cdna4"}
        return _MCPU_TO_ISA.get(self.mcpu, "cdna3")

    @property
    def lds_swizzle(self):
        from aster.layout import Swizzle

        return Swizzle(
            bits=self.lds_swizzle_bits,
            base=self.lds_swizzle_base,
            shift=self.lds_swizzle_shift,
        )

    def tile_elements(self, mfma_shape: list[int]) -> list[int]:
        """[M, N, K] elements per tile = mfma_shape[d] * tile_mult[d]."""
        return [s * m for s, m in zip(mfma_shape, self.tile_mult)]

    @staticmethod
    def default_tile_elements(
        mfma_shape: list[int], tile_mult: list[int] | None = None
    ) -> list[int]:
        """Tile elements for default mapping: mfma_shape * tile_mult."""
        tm = tile_mult or [1, 1, 2]
        return [s * m for s, m in zip(mfma_shape, tm)]

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

    # --- Resource estimates ---
    # VGPRs per dwordx4 global load result.
    VGPRS_PER_LOAD = 4
    # AGPRs per MFMA C accumulator fragment (f32 4x1).
    AGPRS_PER_MFMA_RESULT = 4

    def estimated_agprs(self) -> int:
        return (
            self.num_tiles_per_wave[DIM_M]
            * self.num_tiles_per_wave[DIM_N]
            * self.AGPRS_PER_MFMA_RESULT
        )

    def _operand_vgpr_depth(self, s: dict[str, int], prefix: str) -> int:
        """Live-range depth for one operand's global load buffers.

        LDS path: max(LDS_WRITE - GLOBAL_LOAD, COMPUTE - LDS_READ)
        Direct path: COMPUTE - GLOBAL_LOAD
        """
        load = s[f"{prefix}_LOAD"]
        if prefix == "B" and self.direct_b or prefix == "A" and self.direct_a:
            return s["COMPUTE"] - load
        return max(
            s[f"{prefix}_LDS_WRITE"] - load, s["COMPUTE"] - s[f"{prefix}_LDS_READ"]
        )

    def _coop_tiles_per_wave(self) -> tuple[int, int]:
        """Per-wave cooperative load tile counts for A and B."""
        import math

        twg = self.num_tiles_per_workgroup
        nw = self.num_waves
        kt = self.num_tiles_per_wave[DIM_K]

        def _split(num_tiles, num_waves, kt):
            ws = min(num_tiles, num_waves)
            wk = max(1, math.floor(num_waves / ws))
            return math.ceil(num_tiles / ws) * math.ceil(kt / wk)

        return _split(twg[DIM_M], nw, kt), _split(twg[DIM_N], nw, kt)

    def estimated_vgprs(self) -> int:
        mt, nt, kt = self.num_tiles_per_wave
        s = self._pipeline_stage_dict()
        a_depth = self._operand_vgpr_depth(s, "A")
        b_depth = self._operand_vgpr_depth(s, "B")
        vpl = self.VGPRS_PER_LOAD
        coop_a, coop_b = self._coop_tiles_per_wave()
        a_load = coop_a * max(1, a_depth) * vpl
        a_lds_read = mt * kt * vpl
        b_load = (kt * nt if self.direct_b else coop_b) * max(1, b_depth) * vpl
        b_split = kt * nt * vpl
        overhead = 30 if self.direct_b else 10
        return a_load + a_lds_read + b_load + b_split + overhead

    def lds_bytes(self, tile_bytes: int = 1024, **overrides) -> int:
        """LDS budget estimate for the pre-compile resource filter.

        Uses self.lds_at_write and self.dealloc_at_read as defaults;
        callers can override via keyword args.

        LDS is live from alloc to dealloc.

        Alloc stage:
        - lds_at_write=False: both A and B alloc at min(A_LOAD, B_LOAD)
          (the builder groups them to keep the multi-buffer allocator happy).
        - lds_at_write=True:  each alloc at its own WRITE stage.

        Dealloc stage:
        - dealloc_at_read=True:  dealloc at LDS_READ  (test_102 Python builder).
        - dealloc_at_read=False: dealloc at COMPUTE    (test_001 MLIR templates).

        direct_b: B uses no LDS at all.
        """
        _lds_at_write = overrides.get("lds_at_write", self.lds_at_write)
        _dealloc_at_read = overrides.get("dealloc_at_read", self.dealloc_at_read)
        kt = self.num_tiles_per_wave[DIM_K]
        s = self._pipeline_stage_dict()
        if _lds_at_write:
            a_alloc = s["A_LDS_WRITE"]
            b_alloc = s["B_LDS_WRITE"]
        else:
            # Both allocated at the earliest LOAD stage (matches builder code).
            early = min(s["A_LOAD"], s["B_LOAD"]) if not self.direct_b else s["A_LOAD"]
            a_alloc = early
            b_alloc = early
        a_dealloc = s["A_LDS_READ"] if _dealloc_at_read else s["COMPUTE"]
        a_lds_depth = max(1, a_dealloc - a_alloc + 1)
        a_lds = a_lds_depth * self.num_tiles_per_workgroup[DIM_M] * kt * tile_bytes
        if self.direct_b:
            b_lds = 0
        else:
            b_dealloc = s["B_LDS_READ"] if _dealloc_at_read else s["COMPUTE"]
            b_lds_depth = max(1, b_dealloc - b_alloc + 1)
            b_lds = b_lds_depth * kt * self.num_tiles_per_workgroup[DIM_N] * tile_bytes
        return a_lds + b_lds

    def simd_occupancy(self) -> int:
        import math

        return self.num_wg_per_cu * math.ceil(self.num_waves / 4)


# -----------------------------------------------------------------------------
# OperandLayout -- tile geometry and coord layouts at a given MFMA multiplicity
# -----------------------------------------------------------------------------


class OperandLayout:
    """Per-operand geometry.

    The memory_layout must be flat (non-nested) for now.
    """

    def __init__(
        self,
        memory_layout: Layout,
        logical_dims: list[int],
        elt_bytes: int,
        mfma_shape: list[int],
    ):
        assert memory_layout.is_flat, (
            f"OperandLayout requires flat memory_layout, got {memory_layout!r}"
        )
        self.memory_layout = memory_layout
        self.logical_dims = logical_dims
        self.elt_bytes = elt_bytes
        self.mfma_shape = mfma_shape

    @property
    def rank(self) -> int:
        """Number of logical dimensions."""
        return len(self.memory_layout)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of this operand (sizes from the memory layout)."""
        return (
            self.memory_layout.sizes
            if isinstance(self.memory_layout.sizes, tuple)
            else (self.memory_layout.sizes,)
        )

    @property
    def strides(self) -> tuple[int, ...]:
        """Byte strides from the memory layout."""
        return (
            self.memory_layout.strides
            if isinstance(self.memory_layout.strides, tuple)
            else (self.memory_layout.strides,)
        )

    @property
    def num_elements(self) -> int:
        """Total number of elements."""
        return self.memory_layout.size()


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
        tile_k = self.mapping.tile_elements(self.spec.mfma_shape)[DIM_K]
        return self.gemm_size[DIM_K] // (
            self.mapping.num_tiles_per_wave[DIM_K] * tile_k
        )

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
                r"(_ldsw)?"
                r"(_nosetprio)?"
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
            ldsw,
            nosetprio,
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
            lds_at_write=ldsw is not None,
            set_mfma_priority=nosetprio is None,
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
        ldswrite = "_ldsw" if self.mapping.lds_at_write else ""
        nosetprio = "" if self.mapping.set_mfma_priority else "_nosetprio"
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
            f"{wgcu}{lcm}{um}{peel}{llsched}{hoistwait}{ldswrite}{nosetprio}{suffix}"
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
