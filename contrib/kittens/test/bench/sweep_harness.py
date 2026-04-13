# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep grid framework and shared GEMM sweep infrastructure.

SweepGrid: composable search space with hierarchical pruning.
Shared constants, derivation helpers, resource checks, and CLI args
for 16x16 MFMA GEMM benchmark sweeps.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from bench_harness import check_numpy_blas, detect_num_gpus, verify_on_gpus, _save_tmpfile
from kittens.gemm_config import GemmMappingSpec, WeakScaledMappedGemmInstance

if TYPE_CHECKING:
    from aster.compiler.metadata import KernelResources


# -- Sweep grid framework ---------------------------------------------------


class SweepAxis:
    """One dimension of a sweep search space."""

    __slots__ = ("name", "values")

    def __init__(self, name: str, values: Iterable[Any]):
        self.name = name
        self.values = list(values)


class SweepFilter:
    """Constraint on a combination of axes, applied as a filter."""

    __slots__ = ("deps", "check")

    def __init__(self, deps: tuple[str, ...], check: Callable[[dict[str, Any]], bool]):
        self.deps = deps
        self.check = check


class SweepGrid:
    """Composable sweep search space with hierarchical pruning.

    Build a grid by chaining ``.axis()`` and ``.filter()`` calls, set the
    instance builder with ``.build_with()``, then call ``.generate()``::

        grid = (SweepGrid()
            .axis("waves_m", [1, 2])
            .axis("waves_n", [1, 2])
            .axis("twg_m", range(2, 9))
            .filter("waves_m", "twg_m", check=lambda d: d["twg_m"] % d["waves_m"] == 0)
            .build_with(lambda d: MyInstance(...))
        )
        instances, total = grid.generate(pins={"waves_m": 2}, sample_size=100)

    Axes are enumerated in insertion order. Filters are applied at the
    shallowest nesting level where all their deps are bound, so ordering
    high-selectivity axes first gives better pruning.
    """

    def __init__(self) -> None:
        self._axes: list[SweepAxis] = []
        self._filters: list[SweepFilter] = []
        self._builder: Optional[Callable[[dict[str, Any]], Any]] = None

    def axis(self, name: str, values: Iterable[Any]) -> "SweepGrid":
        """Add a sweep axis."""
        self._axes.append(SweepAxis(name, values))
        return self

    def filter(self, *deps: str, check: Callable[[dict[str, Any]], bool]) -> "SweepGrid":
        """Add a constraint filter.

        ``deps`` are axis names it reads.
        """
        self._filters.append(SweepFilter(deps, check))
        return self

    def build_with(self, fn: Callable[[dict[str, Any]], Any]) -> "SweepGrid":
        """Set the function that builds an instance from a config dict."""
        self._builder = fn
        return self

    def generate(
        self,
        pins: Optional[dict[str, Any]] = None,
        sample_size: int = 3000,
        stratification_key: Optional[Callable[[dict[str, Any]], Hashable]] = None,
        priority_fn: Optional[Callable[[dict[str, Any]], float]] = None,
        extra_eligible: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[Any], int]:
        """Enumerate configs with hierarchical pruning, then sample.

        Args:
            pins: Dict of {axis_name: value} to fix specific axes.
            sample_size: Max configs to return (0 = all).
            stratification_key: Optional callable(config_dict) -> hashable.
                When provided, sampling is stratified: each stratum gets
                an equal share of the sample budget.
            priority_fn: Optional callable(config_dict) -> float.
                When provided, configs are sorted by priority (highest first)
                before sampling. Top configs are selected instead of random.
            extra_eligible: Optional list of axis-level config dicts to add
                to the eligible pool before sampling. These bypass grid
                enumeration (useful for seeded configs like weak-scaled
                variants that may not be reachable through the normal axes).
                Deduplicated against the enumerated pool by dict equality.

        Returns:
            (instances, total_eligible) tuple.
        """
        import random

        pins = pins or {}
        axis_names = [a.name for a in self._axes]

        # Fail loudly if a pin asks for a value that isn't in its axis --
        # silent "0 eligible" is indistinguishable from a filter rejecting
        # everything and has caused at least one debugging detour.
        for axis in self._axes:
            if axis.name in pins and pins[axis.name] not in axis.values:
                raise ValueError(
                    f"pin {axis.name}={pins[axis.name]!r} is not in the axis "
                    f"values {sorted(axis.values)!r}. Check for --mcpu/-m vs "
                    f"-n typos or CDNA3-vs-CDNA4 axis pruning."
                )

        level_filters: list[list[SweepFilter]] = []
        for i in range(len(self._axes)):
            bound_set = set(axis_names[: i + 1])
            prev_set = set(axis_names[:i])
            applicable = [f for f in self._filters if set(f.deps) <= bound_set and not set(f.deps) <= prev_set]
            level_filters.append(applicable)

        eligible: list[dict[str, Any]] = []
        self._enumerate(0, {}, pins, level_filters, eligible)

        if extra_eligible:
            # Dedup against enumerated pool via a frozen fingerprint of dict items.
            seen = {tuple(sorted(c.items())) for c in eligible}
            for extra in extra_eligible:
                key = tuple(sorted(extra.items()))
                if key not in seen:
                    eligible.append(extra)
                    seen.add(key)

        total = len(eligible)

        if sample_size > 0 and total > sample_size:
            if priority_fn is not None:
                # Weighted sampling: configs with higher scores are more
                # likely to be selected, but all configs have a chance.
                scores = [max(priority_fn(c), 0.01) for c in eligible]
                eligible = random.choices(eligible, weights=scores, k=sample_size)
            elif stratification_key is not None:
                eligible = _stratified_sample(eligible, stratification_key, sample_size)
            else:
                eligible = random.sample(eligible, sample_size)

        assert self._builder is not None, "call build_with() before generate()"
        instances = [self._builder(cfg) for cfg in eligible]
        print(f"Total: {total:,} eligible, {len(instances):,} selected")
        return instances, total

    def _enumerate(
        self,
        depth: int,
        bound: dict[str, Any],
        pins: dict[str, Any],
        level_filters: list[list[SweepFilter]],
        out: list[dict[str, Any]],
    ) -> None:
        if depth == len(self._axes):
            out.append(dict(bound))
            return
        axis = self._axes[depth]
        filters = level_filters[depth]
        for val in axis.values:
            if axis.name in pins and pins[axis.name] != val:
                continue
            bound[axis.name] = val
            if all(f.check(bound) for f in filters):
                self._enumerate(depth + 1, bound, pins, level_filters, out)
            del bound[axis.name]


def _stratified_sample(
    configs: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Hashable],
    n: int,
) -> list[dict[str, Any]]:
    """Sample n configs with equal representation per stratum."""
    import random

    strata: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for cfg in configs:
        strata[key_fn(cfg)].append(cfg)
    per_stratum = max(n // len(strata), 1)
    result: list[dict[str, Any]] = []
    for key in sorted(strata):
        s = strata[key]
        if len(s) > per_stratum:
            s = random.sample(s, per_stratum)
        result.extend(s)
    if len(result) > n:
        result = random.sample(result, n)
    return result


def add_scheduling_axes(grid: SweepGrid, unroll_multipliers: Optional[Sequence[int]] = None) -> SweepGrid:
    """Add the common scheduling flag axes shared across sweeps.

    Adds: lcm_unroll, unroll_mult, epilogue_peeling, ll_sched, hoist_wait.
    Also adds the constraint that unroll_mult > 1 requires lcm_unroll=True.
    """
    if unroll_multipliers is None:
        unroll_multipliers = [1, 2, 3]
    grid.axis("lcm_unroll", [True, False])
    grid.axis("unroll_mult", unroll_multipliers)
    grid.axis("epilogue_peeling", [True, False])
    grid.axis("ll_sched", [True, False])
    grid.axis("hoist_wait", [True, False])
    grid.filter("lcm_unroll", "unroll_mult", check=lambda d: d["lcm_unroll"] or d["unroll_mult"] == 1)
    return grid


# -- GPU hardware constants --------------------------------------------------


class GpuHwConstants:
    """Hardware constants for the sweep resource filter.

    Populated from the static arch table in ``aster.core.target`` so the
    eligible search space is deterministic across hosts.
    """

    __slots__ = ("vgprs_per_simd", "max_vgprs", "max_agprs", "lds_per_cu", "vgpr_granule", "num_simds", "mcpu")

    def __init__(
        self,
        vgprs_per_simd: int,
        max_vgprs: int,
        max_agprs: int,
        lds_per_cu: int,
        vgpr_granule: int,
        num_simds: int,
        mcpu: str,
    ):
        self.vgprs_per_simd = vgprs_per_simd
        self.max_vgprs = max_vgprs
        self.max_agprs = max_agprs
        self.lds_per_cu = lds_per_cu
        self.vgpr_granule = vgpr_granule
        self.num_simds = num_simds
        self.mcpu = mcpu


def hw_for_target(mcpu: str) -> GpuHwConstants:
    """Return sweep-filter HW constants for ``mcpu`` from the static arch table."""
    from aster.core.target import Target

    t = Target.from_mcpu(mcpu)
    return GpuHwConstants(
        vgprs_per_simd=t.vgprs_per_simd,
        max_vgprs=t.max_vgprs,
        max_agprs=t.max_agprs,
        lds_per_cu=t.lds_per_cu,
        vgpr_granule=t.vgpr_alloc_granule,
        num_simds=t.num_simds,
        mcpu=t.mcpu,
    )


# -- Resource checks ---------------------------------------------------------


def _estimated_resources(
    mapping: GemmMappingSpec,
    vgpr_headroom: float,
    vgpr_overhead: int,
) -> "KernelResources":
    from aster.compiler.metadata import KernelResources

    return KernelResources(
        vgpr_count=int(mapping.estimated_vgprs() * vgpr_headroom) + vgpr_overhead,
        agpr_count=mapping.estimated_agprs(),
        lds_bytes=mapping.lds_bytes(),
    )


def passes_resource_check(
    mapping: GemmMappingSpec,
    hw: GpuHwConstants,
    vgpr_headroom: float = 1.2,
    vgpr_overhead: int = 16,
) -> bool:
    """Pre-compile resource filter, must reflect the desired WG occupancy set on the sweep axis."""
    res = _estimated_resources(mapping, vgpr_headroom, vgpr_overhead)
    violations = res.check_occupancy(
        mapping.num_threads,
        mcpu=hw.mcpu,
        num_wg_per_cu=mapping.num_wg_per_cu,
    )
    return not violations


def fits_on_cu_post_compile(
    cfg: "WeakScaledMappedGemmInstance",
    res: "KernelResources",
) -> Optional[str]:
    """Post-compile occupancy check using actual register + LDS counts from ASM.

    Returns ``None`` if the config fits, or a one-line reason string otherwise.
    """
    violations = res.check_occupancy(
        cfg.num_threads,
        mcpu=cfg.mcpu,
        num_wg_per_cu=getattr(cfg, "num_wg_per_cu", 1),
    )
    if not violations:
        return None
    mm: GemmMappingSpec = cfg.mapping
    est_lds = mm.lds_bytes()
    est_v = mm.estimated_vgprs()
    est_a = mm.estimated_agprs()
    return (
        f"occupancy: est(lds={est_lds}, v={est_v}, a={est_a}) "
        f"vs actual(lds={res.lds_bytes}, v={res.vgpr_count}, a={res.agpr_count}) -- "
        + "; ".join(violations)
    )


def add_resource_filter(
    grid: SweepGrid,
    hw: GpuHwConstants,
    mapping_builder: Callable[[dict[str, Any]], GemmMappingSpec],
    deps: tuple[str, ...] = (),
) -> None:
    """Add a resource-check filter to a SweepGrid."""

    def _check(d: dict[str, Any]) -> bool:
        return passes_resource_check(mapping_builder(d), hw)

    grid.filter(*deps, check=_check)


# -- Shared GEMM sweep constants ---------------------------------------------

# Workgroup base grid: M and N workgroup counts are multiples of this.
WG_BASE = (19, 16)

# Wave configs: valid (waves_m, waves_n) pairs where total is a multiple of 4.
_WAVE_BASES = [(1, 4), (2, 2), (4, 1)]
WAVE_CONFIGS = sorted(
    {
        (bm * k1, bn * k2)
        for bm, bn in _WAVE_BASES
        for k1 in range(1, 7)
        for k2 in range(1, 7)
        if bm * k1 <= 6 and bn * k2 <= 8 and bm * k1 * bn * k2 <= 16 and (bm * k1 * bn * k2) % 4 == 0
    }
)


# Default problem dimension when not pinned via --m/--n/--k.
DEFAULT_DIM = 4096


def dim_values(
    pin: int | None,
    tile_size: int,
    default: int = DEFAULT_DIM,
    min_val: int = 1024,
    max_val: int = 24576,
    step: int = 128,
) -> list[int]:
    """Return the value set for a problem dimension axis.

    This is the single mechanism for all size scenarios:
      - pin is an int: returns [pin] (user pinned this dimension)
      - pin is None: returns multiples of step in [min_val, max_val]
        that are also divisible by tile_size

    Used by --m/--n/--k, --size, and default sweep modes.
    """
    if pin is not None:
        assert pin % tile_size == 0, f"{pin} not divisible by tile_size={tile_size}"
        return [pin]
    # Sweep: multiples of step that are also multiples of tile_size.
    from math import gcd

    effective_step = step * tile_size // gcd(step, tile_size)  # lcm
    start = max(min_val, effective_step)
    return list(range(start, max_val + 1, effective_step))


# -- Derivation helpers (occupancy -> WG sizing) -----------------------------


def wps(d: dict[str, Any], hw: GpuHwConstants) -> int:
    """Waves per SIMD for a config dict with waves_m, waves_n."""
    return (d["waves_m"] * d["waves_n"] + hw.num_simds - 1) // hw.num_simds


def nwgcu(d: dict[str, Any], hw: GpuHwConstants) -> int:
    """Workgroups per CU from occupancy target."""
    return d["occ"] // wps(d, hw)


def wg_m(d: dict[str, Any], hw: GpuHwConstants) -> int:
    """M-dimension workgroup count (WG_BASE[0] * nwgcu)."""
    return WG_BASE[0] * nwgcu(d, hw)


def wg_n(d: dict[str, Any]) -> int:
    """N-dimension workgroup count (WG_BASE[1] * n_mult)."""
    return WG_BASE[1] * d["n_mult"]


def resolve_derived_pins(pins: dict[str, Any]) -> dict[str, Any]:
    """Convert derived-value pins (wg_m, wg_n, occ) to axis-level pins.

    Pins are passed through as ``_wg_m`` / ``_wg_n`` keys; the bench
    script is responsible for adding a filter that honors them. This
    allows different benches to compute ``wg_m`` differently (target-size
    decomposition vs base-grid derivation).
    """
    if not pins:
        return pins
    out = dict(pins)
    if "wg_m_pin" in out:
        out["_wg_m"] = out.pop("wg_m_pin")
    if "wg_n_pin" in out:
        out["_wg_n"] = out.pop("wg_n_pin")
    if "occ_pin" in out:
        out["occ"] = out.pop("occ_pin")
    return out


def add_gemm_sweep_axes(
    grid: SweepGrid,
    hw: GpuHwConstants,
    tile_elements: list[int],
    *,
    target_m: int | None = DEFAULT_DIM,
    target_n: int | None = DEFAULT_DIM,
    target_k: int | None = DEFAULT_DIM,
) -> None:
    """Add problem-size + geometry + scheduling axes and constraints.

    Problem dimensions (M, N, K) are controlled via dim_values():
      - int: pin to that value
      - None: sweep multiples of 128 in [1024, 24576]

    ``tile_elements`` is [tile_m, tile_n, tile_k] from
    ``GemmMappingSpec.tile_elements(spec.mfma_shape)``.

    The CDNA3-vs-CDNA4 axis pruning is driven by ``hw.mcpu`` -- callers
    must use ``hw_for_target(args.mcpu)`` so the search space matches the
    compile target, not the host.
    """
    from kittens_helpers import PIPELINE_STRATEGIES as PS

    tile_m, tile_n, tile_k = tile_elements

    _twg_m_vals = sorted({mw * mm for (mw, _), mm in itertools.product(WAVE_CONFIGS, range(1, 6))})
    _twg_n_vals = sorted({nw * nm for (_, nw), nm in itertools.product(WAVE_CONFIGS, range(1, 11))})

    # Problem-size axes (placed first for early pruning).
    grid.axis("target_M", dim_values(target_m, tile_m))
    grid.axis("target_N", dim_values(target_n, tile_n))
    grid.axis("target_K", dim_values(target_k, tile_k))

    # Prune the search space for CDNA3 (smaller LDS + narrower axis ranges).
    _cdna3 = hw.mcpu in ("gfx940", "gfx942")

    grid.axis("waves_m", sorted({m for m, _ in WAVE_CONFIGS if not _cdna3 or m <= 2}))
    grid.axis("waves_n", sorted({n for _, n in WAVE_CONFIGS}))
    grid.axis("occ", [1, 2, 3] if _cdna3 else [1, 2, 3, 4])
    grid.axis("n_mult", list(range(1, 11)))
    grid.axis("twg_m", _twg_m_vals)
    grid.axis("twg_n", [v for v in _twg_n_vals if not _cdna3 or v >= 6])
    grid.axis("twg_k", [1, 2] if _cdna3 else list(range(1, 9)))
    grid.axis("ps", list(range(1, 9)) if _cdna3 else list(range(1, 11)))
    add_scheduling_axes(grid, unroll_multipliers=[1, 2, 3])

    # Wave config validity.
    _wc = set(WAVE_CONFIGS)
    grid.filter("waves_m", "waves_n", check=lambda d: (d["waves_m"], d["waves_n"]) in _wc)
    grid.filter("waves_m", "waves_n", "occ", check=lambda d: d["occ"] % wps(d, hw) == 0)
    grid.filter("waves_m", "twg_m", check=lambda d: d["twg_m"] % d["waves_m"] == 0)
    grid.filter("waves_n", "twg_n", check=lambda d: d["twg_n"] % d["waves_n"] == 0)

    # Size decomposition: target == wg * twg * tile_size, must divide evenly.
    grid.filter(
        "target_M",
        "twg_m",
        check=lambda d, t=tile_m: d["target_M"] % (d["twg_m"] * t) == 0,
    )
    grid.filter(
        "target_N",
        "twg_n",
        check=lambda d, t=tile_n: d["target_N"] % (d["twg_n"] * t) == 0,
    )

    # K decomposition: target_K = k_iters * twg_k * tile_k.
    # k_iters must exceed the deepest pipeline stage.
    _ps_max = {ps: max(stages.values()) for ps, stages in PS.items()}
    grid.filter(
        "target_K",
        "twg_k",
        check=lambda d, t=tile_k: d["target_K"] % (d["twg_k"] * t) == 0,
    )
    grid.filter(
        "target_K",
        "twg_k",
        "ps",
        check=lambda d, t=tile_k, pm=_ps_max: d["target_K"] // (d["twg_k"] * t) > pm[d["ps"]],
    )


# Standard pin mapping for CLI args -> axis names.
GEMM_SWEEP_PIN_MAP = {
    "num_workgroups_m": "wg_m_pin",
    "num_workgroups_n": "wg_n_pin",
    "waves_per_wg_m": "waves_m",
    "waves_per_wg_n": "waves_n",
    "tiles_per_wg_m": "twg_m",
    "tiles_per_wg_n": "twg_n",
    "tiles_per_wg_k": "twg_k",
    "pipeline_strategy": "ps",
    "desired_simd_occupancy": "occ_pin",
    "unroll_multiplier": "unroll_mult",
    "lcm_unroll": "lcm_unroll",
    "epilogue_peeling": "epilogue_peeling",
    "ll_sched": "ll_sched",
    "hoist_wait": "hoist_wait",
    "set_mfma_priority": "set_mfma_priority",
}


def is_label(s: str) -> bool:
    """Check if a string looks like a serialized config label."""
    return s.startswith("m") and "x" in s and "_wg" in s


def add_size_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add --m, --n, --k, --size CLI args shared across bench scripts."""
    parser.add_argument("--m", type=int, default=None, help=f"M dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--n", type=int, default=None, help=f"N dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--k", type=int, default=None, help=f"K dimension (default: {DEFAULT_DIM})")
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        metavar="MxNxK",
        help="Pin all dimensions (exclusive with --m/--n/--k)",
    )


def parse_size_args(args, parser) -> tuple[int, int, int]:
    """Resolve --size vs --m/--n/--k into (target_m, target_n, target_k).

    Unpinned dimensions default to DEFAULT_DIM.
    """
    has_mnk = any(getattr(args, a, None) is not None for a in ("m", "n", "k"))
    if args.size and has_mnk:
        parser.error("--size is exclusive with --m/--n/--k")
    if args.size:
        parts = args.size.split("x")
        if len(parts) != 3:
            parser.error("--size must be MxNxK (e.g., 2432x12288x4096)")
        return int(parts[0]), int(parts[1]), int(parts[2])
    return (
        getattr(args, "m", None) or DEFAULT_DIM,
        getattr(args, "n", None) or DEFAULT_DIM,
        getattr(args, "k", None) or DEFAULT_DIM,
    )


def apply_wg_pin_filters(
    grid: SweepGrid,
    pins: dict | None,
    tile_m: int,
    tile_n: int,
) -> None:
    """Honor _wg_m / _wg_n pins via target-size decomposition filters.

    The filters enforce ``target_M == wg_m * twg_m * tile_m`` (and the N
    analogue). The pins are consumed (popped) from the ``pins`` dict.
    Works for any wg_m/wg_n value regardless of WG_BASE.
    """
    if not pins:
        return
    if "_wg_m" in pins:
        target = pins.pop("_wg_m")
        grid.filter(
            "target_M",
            "twg_m",
            check=lambda d, t=target, tm=tile_m: d["target_M"] == t * d["twg_m"] * tm,
        )
    if "_wg_n" in pins:
        target = pins.pop("_wg_n")
        grid.filter(
            "target_N",
            "twg_n",
            check=lambda d, t=target, tn=tile_n: d["target_N"] == t * d["twg_n"] * tn,
        )


# -- Sweep verification -----------------------------------------------------


def verify_top_configs(
    results: list,
    hsaco_paths: dict,
    repro_cmd_fn: Callable,
    *,
    mcpu: str,
    top_n: int = 100,
    num_gpus: Optional[int] = None,
    label: str = "",
) -> None:
    """Phase 3: Verify top N configs for correctness using subprocess isolation."""
    if not results:
        return
    if num_gpus is None:
        num_gpus = detect_num_gpus(mcpu)
    if num_gpus == 0:
        print("\nNo GPUs detected -- skipping correctness verification.")
        return
    top = results[:top_n]
    to_verify = [c for c, *_ in top if c.label in hsaco_paths]
    if not to_verify:
        return
    print(f"\n--- Phase 3: Correctness ({len(to_verify)} configs, {num_gpus} GPU(s)) ---")
    check_numpy_blas(label="correctness")
    passed, errors = verify_on_gpus(to_verify, hsaco_paths, num_gpus)
    print(f"\nCorrectness: {passed}/{len(to_verify)} passed", end="")
    if errors:
        cfg_map = {c.label: c for c in to_verify}
        enriched = []
        for e in errors:
            lbl = e.split(":")[0].strip()
            repro = ""
            if lbl in cfg_map:
                try:
                    repro = f"\n  repro: {repro_cmd_fn(cfg_map[lbl])}"
                except Exception:
                    pass
            enriched.append(f"{e}{repro}")
        prefix = f"bench_verify_{label}_" if label else "bench_verify_"
        path = _save_tmpfile(prefix, enriched)
        print(f", {len(errors)} FAILED (details in {path})")
    else:
        print(" -- all correct")


# -- Common sweep CLI args ---------------------------------------------------


def add_geometry_pin_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared geometry pinning CLI args."""
    parser.add_argument("--num-workgroups-m", type=int, help="Pin workgroups along M")
    parser.add_argument("--num-workgroups-n", type=int, help="Pin workgroups along N")
    parser.add_argument("--waves-per-wg-m", type=int, help="Pin waves per WG along M")
    parser.add_argument("--waves-per-wg-n", type=int, help="Pin waves per WG along N")
    parser.add_argument("--tiles-per-wg-m", type=int, help="Pin tiles per workgroup along M")
    parser.add_argument("--tiles-per-wg-n", type=int, help="Pin tiles per workgroup along N")
    parser.add_argument("--tiles-per-wg-k", type=int, help="Pin tiles per wave along K")
    parser.add_argument(
        "--pipeline-strategy",
        type=int,
        default=None,
        choices=range(0, 11),
        metavar="{0..10}",
        help="Pin pipeline strategy",
    )
    parser.add_argument("--lcm-unroll", action=argparse.BooleanOptionalAction, default=None, help="Pin LCM unrolling")
    parser.add_argument("--unroll-multiplier", type=int, default=None, help="Pin unroll multiplier")
    parser.add_argument(
        "--epilogue-peeling", action=argparse.BooleanOptionalAction, default=None, help="Pin epilogue peeling"
    )
    parser.add_argument(
        "--ll-sched", action=argparse.BooleanOptionalAction, default=None, help="Pin low-level scheduler"
    )
    parser.add_argument(
        "--hoist-wait", action=argparse.BooleanOptionalAction, default=None, help="Pin hoist iter_arg waits"
    )
    parser.add_argument("--desired-simd-occupancy", type=int, default=None, help="Pin SIMD occupancy")
    parser.add_argument("--direct-b", action=argparse.BooleanOptionalAction, default=None, help="B via preshuffle")
