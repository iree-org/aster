# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep grid framework."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Optional

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
    """Constraint over axes in `deps`.

    Named filters can be selectively retained per tier; anonymous
    filters always apply.
    """

    __slots__ = ("deps", "check", "name")

    def __init__(
        self,
        deps: tuple[str, ...],
        check: Callable[[dict[str, Any]], bool],
        name: Optional[str] = None,
    ):
        self.deps = deps
        self.check = check
        self.name = name


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
        instances, cfgs, total = grid.generate(sample_size=100)

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

    def filter(
        self,
        *deps: str,
        check: Callable[[dict[str, Any]], bool],
        name: Optional[str] = None,
    ) -> "SweepGrid":
        """Add a constraint filter.

        ``deps`` are axis names it reads. ``name`` is an optional stable
        identifier; if set, callers (e.g. `bench_tier_schedule.apply_tier_overrides`)
        can include / exclude this filter by name when building per-tier grids.

        Each ``dep`` must name a registered axis. An unknown dep would make
        ``set(deps) <= bound_set`` permanently false, so the filter would be
        silently dropped from enumeration. Register all axes (including
        ``add_gemm_sweep_axes`` / ``add_scheduling_axes``) before calling
        ``filter()``.
        """
        known = {a.name for a in self._axes}
        missing = [d for d in deps if d not in known]
        if missing:
            tag = f" (filter {name!r})" if name else ""
            raise ValueError(f"SweepGrid.filter: unknown axis dep(s) {missing}{tag}. Registered axes: {sorted(known)}")
        self._filters.append(SweepFilter(deps, check, name))
        return self

    def build_with(self, fn: Callable[[dict[str, Any]], Any]) -> "SweepGrid":
        """Set the function that builds an instance from a config dict."""
        self._builder = fn
        return self

    def filter_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self._filters if f.name is not None)

    def axis_values(self) -> dict[str, list[Any]]:
        return {axis.name: list(axis.values) for axis in self._axes}

    def restrict_axes(self, overrides: dict[str, list[Any]]) -> None:
        """Replace value list for each axis in `overrides`."""
        for axis in self._axes:
            if axis.name in overrides:
                axis.values = list(overrides[axis.name])

    def retain_filters(self, allowed_names: set[str]) -> None:
        """Drop named filters not in `allowed_names`.

        Anonymous filters survive.
        """
        self._filters = [f for f in self._filters if f.name is None or f.name in allowed_names]

    def generate(
        self,
        sample_size: int = 3000,
        stratification_key: Optional[Callable[[dict[str, Any]], Hashable]] = None,
        extra_eligible: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[Any], list[dict[str, Any]], int]:
        """Enumerate, sample, build. Returns (instances, cfgs, total_eligible).

        `cfgs[i]` is the axis dict that produced `instances[i]` -- callers
        chaining tiers read winner axes from `cfgs`. `extra_eligible` configs
        bypass enumeration and survive sampling.
        """
        import random

        axis_names = [a.name for a in self._axes]

        level_filters: list[list[SweepFilter]] = []
        for i in range(len(self._axes)):
            bound_set = set(axis_names[: i + 1])
            prev_set = set(axis_names[:i])
            applicable = [f for f in self._filters if set(f.deps) <= bound_set and not set(f.deps) <= prev_set]
            level_filters.append(applicable)

        eligible: list[dict[str, Any]] = []
        self._enumerate(0, {}, level_filters, eligible)

        # extras matching an enumerated config are moved here so they survive sampling.
        extras_protected: list[dict[str, Any]] = []
        if extra_eligible:
            all_filters = [f for level in level_filters for f in level]
            enum_index: dict[tuple, int] = {tuple(sorted(c.items())): i for i, c in enumerate(eligible)}
            indices_to_remove: set[int] = set()
            seen: set[tuple] = set()
            n_dropped_filter = 0
            for extra in extra_eligible:
                key = tuple(sorted(extra.items()))
                if key in seen:
                    continue
                if any(not f.check(extra) for f in all_filters if all(d in extra for d in f.deps)):
                    n_dropped_filter += 1
                    continue
                if key in enum_index:
                    indices_to_remove.add(enum_index[key])
                extras_protected.append(extra)
                seen.add(key)
            if indices_to_remove:
                eligible = [c for i, c in enumerate(eligible) if i not in indices_to_remove]
            if n_dropped_filter:
                print(f"  extras: dropped {n_dropped_filter} (filter rejected)")

        total = len(eligible) + len(extras_protected)

        # Sample enumerated only; budget reduced by protected count.
        if sample_size > 0:
            sample_target = max(0, sample_size - len(extras_protected))
            if sample_target == 0:
                eligible = []
            elif len(eligible) > sample_target:
                if stratification_key is not None:
                    eligible = _stratified_sample(eligible, stratification_key, sample_target)
                else:
                    eligible = random.sample(eligible, sample_target)

        eligible = list(eligible) + list(extras_protected)

        assert len({tuple(sorted(c.items())) for c in eligible}) == len(eligible), "sweep produced duplicate configs"

        assert self._builder is not None, "call build_with() before generate()"
        instances = [self._builder(cfg) for cfg in eligible]
        cfgs = list(eligible)

        # Dedup by instance label catches axes that don't reach _build_instance.
        labels = [getattr(inst, "label", None) for inst in instances]
        if labels and all(lbl is not None for lbl in labels):
            seen_labels: set[str] = set()
            unique_instances: list[Any] = []
            unique_cfgs: list[dict[str, Any]] = []
            for lbl, inst, cfg in zip(labels, instances, cfgs):
                if lbl in seen_labels:
                    continue
                seen_labels.add(lbl)
                unique_instances.append(inst)
                unique_cfgs.append(cfg)
            n_collapsed = len(instances) - len(unique_instances)
            if n_collapsed:
                print(
                    f"  dedup: collapsed {n_collapsed} duplicate-by-label sample(s); "
                    f"check for grid axes that do not reach _build_instance"
                )
            instances, cfgs = unique_instances, unique_cfgs
            assert len({inst.label for inst in instances}) == len(instances), (
                "post-build dedup failed: duplicate labels remain"
            )

        print(f"Total: {total:,} eligible, {len(instances):,} selected")
        return instances, cfgs, total

    def _enumerate(
        self,
        depth: int,
        bound: dict[str, Any],
        level_filters: list[list[SweepFilter]],
        out: list[dict[str, Any]],
    ) -> None:
        if depth == len(self._axes):
            out.append(dict(bound))
            return
        axis = self._axes[depth]
        filters = level_filters[depth]
        for val in axis.values:
            bound[axis.name] = val
            if all(f.check(bound) for f in filters):
                self._enumerate(depth + 1, bound, level_filters, out)
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


def add_scheduling_axes(grid: SweepGrid) -> None:
    """Register scheduling-flag axis names (values supplied by tier-1)."""
    for name in (
        "lcm_unroll",
        "epilogue_peeling",
        "ll_sched",
        "hoist_wait",
        "rotate_compute_stage",
    ):
        grid.axis(name, [])


# Sweep-axis -> GemmMappingSpec kwarg. New axes that must reach the mapping
# add an entry here instead of threading a field through every _build_instance.
_SWEEP_TO_MAPPING_KWARG = {
    "lcm_unroll": "lcm_unroll",
    "unroll_factor_multiplier": "unroll_factor_multiplier",
    "epilogue_peeling": "epilogue_peeling",
    "ll_sched": "ll_sched",
    "hoist_wait": "hoist_wait",
    "lds_at_write": "lds_at_write",
    "rotate_compute_stage": "rotate_compute_stage",
}


def mapping_kwargs_from_sweep(d: dict) -> dict:
    """Forward sweep-axis values to GemmMappingSpec kwargs.

    Each bench's _build_instance MUST `**mapping_kwargs_from_sweep(d)`
    so swept axes reach the compiled config; missing axes fall to
    mapping defaults.
    """
    return {mapping_kwarg: d[axis] for axis, mapping_kwarg in _SWEEP_TO_MAPPING_KWARG.items() if axis in d}


# -- GPU hardware constants --------------------------------------------------


class GpuHwConstants:
    """Hardware constants for resource filtering.

    Built via ``hw_for_target(mcpu)``.
    """

    __slots__ = (
        "vgprs_per_simd",
        "max_vgprs",
        "max_agprs",
        "lds_per_cu",
        "vgpr_granule",
        "num_simds",
        "num_cus",
        "mcpu",
    )

    def __init__(
        self,
        *,
        vgprs_per_simd: int,
        max_vgprs: int,
        max_agprs: int,
        lds_per_cu: int,
        vgpr_granule: int,
        num_simds: int,
        num_cus: int,
        mcpu: str,
    ):
        self.vgprs_per_simd = vgprs_per_simd
        self.max_vgprs = max_vgprs
        self.max_agprs = max_agprs
        self.lds_per_cu = lds_per_cu
        self.vgpr_granule = vgpr_granule
        self.num_simds = num_simds
        self.num_cus = num_cus
        self.mcpu = mcpu


# Cross-compile fallback for the CU count when no GPU is present
# (e.g. authoring on macOS). On-GPU runs override this via HIP.
_NUM_CUS_FALLBACK: dict[str, int] = {
    "gfx940": 228,  # MI300A
    "gfx942": 304,  # MI300X
    "gfx950": 256,  # MI350X
    "gfx1201": 64,  # RDNA4 (varies by SKU; conservative default)
}


def hw_for_target(mcpu: str) -> GpuHwConstants:
    """Build hardware constants for `mcpu`.

    `num_cus` from HIP if available, else fallback.
    """
    from aster.core.target import Target

    t = Target.from_mcpu(mcpu)
    num_cus = _NUM_CUS_FALLBACK[mcpu]
    try:
        from aster.core.device import try_query_device

        dp = try_query_device(0)
        if dp is not None and dp.gcn_arch_name == mcpu:
            num_cus = dp.multiprocessor_count
    except ImportError:
        pass
    return GpuHwConstants(
        vgprs_per_simd=t.vgprs_per_simd,
        max_vgprs=t.max_vgprs,
        max_agprs=t.max_agprs,
        lds_per_cu=t.lds_per_cu,
        vgpr_granule=t.vgpr_alloc_granule,
        num_simds=t.num_simds,
        num_cus=num_cus,
        mcpu=mcpu,
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
        mcpu=cfg.mapping.mcpu,
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
        f"vs actual(lds={res.lds_bytes}, v={res.vgpr_count}, a={res.agpr_count}) -- " + "; ".join(violations)
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

    grid.filter(*deps, check=_check, name="passes_resource_check")


# -- Default problem dimension when not pinned via --m/--n/--k -------------

DEFAULT_DIM = 4096


# -- Derivation helpers (occupancy -> WG sizing) -----------------------------


def wps(d: dict[str, Any], hw: GpuHwConstants) -> int:
    """Waves per SIMD for a config dict with waves_m, waves_n."""
    return (d["waves_m"] * d["waves_n"] + hw.num_simds - 1) // hw.num_simds


def nwgcu(d: dict[str, Any], hw: GpuHwConstants) -> int:
    """Workgroups per CU from occupancy target."""
    return d["occ"] // wps(d, hw)


def add_gemm_sweep_axes(grid: SweepGrid) -> None:
    """Register GEMM-sweep axis names as empty placeholders.

    Values come from each bench's `make_tiered_schedule(...).axis_grid` /
    `fixed_axes`. Admissibility filters live in each bench's `_make_grid`.
    """
    for name in (
        "target_M",
        "target_N",
        "target_K",
        "waves_m",
        "waves_n",
        "occ",
        "twg_m",
        "twg_n",
        "twg_k",
        "wg_m",
        "wg_n",
        "ps",
        "unroll_factor_multiplier",
    ):
        grid.axis(name, [])
    add_scheduling_axes(grid)


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
) -> set[str]:
    """Phase 3: verify top N configs for correctness. Returns the set of passing labels."""
    if not results:
        return set()
    if num_gpus is None:
        num_gpus = detect_num_gpus(mcpu)
    if num_gpus == 0:
        print("\nNo GPUs detected -- skipping correctness verification.")
        return set()
    top = results[:top_n]
    to_verify = [c for c, *_ in top if c.label in hsaco_paths]
    if not to_verify:
        return set()
    # Reuse the .hsaco artifacts compiled during the sweep -- no recompilation
    # in phase 3 (verify_on_gpus only consumes hsaco_paths[label] paths).
    print(f"\n--- Phase 3: Correctness ({len(to_verify)} configs, reusing precompiled .hsaco, {num_gpus} GPU(s)) ---")
    check_numpy_blas(label="correctness")
    passed, errors = verify_on_gpus(to_verify, hsaco_paths, num_gpus)
    failed_labels = {e.split(":")[0].strip() for e in errors}
    verified = {c.label for c in to_verify if c.label not in failed_labels}
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
    return verified
