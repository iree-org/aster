"""Sweep grid framework and shared GEMM sweep infrastructure.

SweepGrid: composable search space with hierarchical pruning.
Shared constants, derivation helpers, resource checks, and CLI args
for 16x16 MFMA GEMM benchmark sweeps.
"""

import argparse
import itertools
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from bench_harness import check_numpy_blas, detect_num_gpus, verify_on_gpus, _save_tmpfile
from kittens.gemm_config import GemmMappingSpec


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

        Returns:
            (instances, total_eligible) tuple.
        """
        import random

        pins = pins or {}
        axis_names = [a.name for a in self._axes]

        level_filters: list[list[SweepFilter]] = []
        for i in range(len(self._axes)):
            bound_set = set(axis_names[: i + 1])
            prev_set = set(axis_names[:i])
            applicable = [f for f in self._filters if set(f.deps) <= bound_set and not set(f.deps) <= prev_set]
            level_filters.append(applicable)

        eligible: list[dict[str, Any]] = []
        self._enumerate(0, {}, pins, level_filters, eligible)
        total = len(eligible)

        if sample_size > 0 and total > sample_size:
            if priority_fn is not None:
                eligible.sort(key=priority_fn, reverse=True)
                eligible = eligible[:sample_size]
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
    """Hardware constants for resource filtering, queried at import time."""

    __slots__ = ("vgprs_per_simd", "max_vgprs", "max_agprs", "lds_per_cu", "vgpr_granule", "num_simds")

    def __init__(
        self,
        vgprs_per_simd: int = 512,
        max_vgprs: int = 256,
        max_agprs: int = 256,
        lds_per_cu: int = 65536,
        vgpr_granule: int = 8,
        num_simds: int = 4,
    ):
        self.vgprs_per_simd = vgprs_per_simd
        self.max_vgprs = max_vgprs
        self.max_agprs = max_agprs
        self.lds_per_cu = lds_per_cu
        self.vgpr_granule = vgpr_granule
        self.num_simds = num_simds


def query_gpu_hw() -> GpuHwConstants:
    """Query GPU hardware via HIP, fall back to gfx942 defaults."""
    try:
        from aster.core.device import try_query_device

        dev = try_query_device(0)
    except ImportError:
        dev = None
    if dev is None:
        return GpuHwConstants()
    return GpuHwConstants(
        vgprs_per_simd=dev.vgprs_per_simd,
        max_vgprs=min(dev.vgprs_per_simd, 256),
        max_agprs=min(dev.vgprs_per_simd, 256),
        lds_per_cu=dev.lds_per_cu,
        vgpr_granule=dev.vgpr_alloc_granule,
    )


# -- Resource checks ---------------------------------------------------------


def passes_resource_check(
    mapping: GemmMappingSpec,
    hw: GpuHwConstants,
    vgpr_headroom: float = 1.2,
    vgpr_overhead: int = 16,
) -> bool:
    """Pre-compile resource filter: LDS + VGPR estimates vs hardware limits."""
    nwg = mapping.num_wg_per_cu
    if mapping.lds_bytes() > hw.lds_per_cu // max(nwg, 1):
        return False
    est_v = int(mapping.estimated_vgprs() * vgpr_headroom) + vgpr_overhead
    est_a = mapping.estimated_agprs()
    if est_v > hw.max_vgprs or est_a > hw.max_agprs:
        return False
    combined = est_v + est_a
    if combined > hw.vgprs_per_simd:
        return False
    aligned = ((combined + hw.vgpr_granule - 1) // hw.vgpr_granule) * hw.vgpr_granule
    total_waves = mapping.num_waves * nwg
    if aligned * total_waves * 64 > hw.vgprs_per_simd * hw.num_simds * 64:
        return False
    return True


def fits_on_cu_post_compile(cfg, res) -> bool:
    """Post-compile occupancy check using actual register counts from ASM."""
    violations = res.check_occupancy(
        cfg.num_threads,
        num_wg_per_cu=getattr(cfg, "num_wg_per_cu", 1),
    )
    if violations:
        for v in violations:
            print(f"  OCCUPANCY ERROR [{cfg.label}]: {v}")
        return False
    return True


def add_resource_filter(
    grid: SweepGrid,
    hw: GpuHwConstants,
    mapping_builder: Callable[[dict[str, Any]], Any],
    deps: tuple[str, ...] = (),
) -> None:
    """Add a resource-check filter to a SweepGrid."""

    def _check(d: dict) -> bool:
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

# Minimum problem dimension (M, N, or K) to include in a sweep.
MIN_DIM = 2000

# MFMA tile size (from GemmSpec default mfma_shape[0]).
MFMA_M = 16


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
    """Convert derived-value pins (wg_m, wg_n, occ) to axis-level pins."""
    if not pins:
        return pins
    out = dict(pins)
    if "wg_m_pin" in out:
        val = out.pop("wg_m_pin")
        if val % WG_BASE[0] == 0:
            out["_wg_m"] = val
    if "wg_n_pin" in out:
        val = out.pop("wg_n_pin")
        if val % WG_BASE[1] == 0:
            out["n_mult"] = val // WG_BASE[1]
    if "occ_pin" in out:
        out["occ"] = out.pop("occ_pin")
    return out


def add_gemm_sweep_axes(grid: SweepGrid, hw: GpuHwConstants) -> None:
    """Add the standard GEMM geometry + scheduling axes and constraints.

    Adds: waves_m, waves_n, occ, n_mult, twg_m, twg_n, twg_k, ps, k_factor,
    plus the 5 scheduling flag axes. Also adds all standard geometry constraints.
    """
    from kittens_helpers import PIPELINE_STRATEGIES as PS

    _twg_m_vals = sorted({mw * mm for (mw, _), mm in itertools.product(WAVE_CONFIGS, range(1, 6))})
    _twg_n_vals = sorted({nw * nm for (_, nw), nm in itertools.product(WAVE_CONFIGS, range(1, 11))})

    grid.axis("waves_m", sorted({m for m, _ in WAVE_CONFIGS}))
    grid.axis("waves_n", sorted({n for _, n in WAVE_CONFIGS}))
    grid.axis("occ", [1, 2, 3, 4])
    grid.axis("n_mult", list(range(1, 11)))
    grid.axis("twg_m", _twg_m_vals)
    grid.axis("twg_n", _twg_n_vals)
    grid.axis("twg_k", list(range(1, 9)))
    grid.axis("ps", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    grid.axis("k_factor", [64, 128, 256])
    add_scheduling_axes(grid, unroll_multipliers=[1, 2, 3])

    _wc = set(WAVE_CONFIGS)
    grid.filter("waves_m", "waves_n", check=lambda d: (d["waves_m"], d["waves_n"]) in _wc)
    grid.filter("waves_m", "waves_n", "occ", check=lambda d: d["occ"] % wps(d, hw) == 0)
    grid.filter("waves_m", "twg_m", check=lambda d: d["twg_m"] % d["waves_m"] == 0)
    grid.filter("waves_n", "twg_n", check=lambda d: d["twg_n"] % d["waves_n"] == 0)
    grid.filter(
        "waves_m", "waves_n", "occ", "twg_m",
        check=lambda d: wg_m(d, hw) * d["twg_m"] * MFMA_M >= MIN_DIM,
    )
    grid.filter("n_mult", "twg_n", check=lambda d: wg_n(d) * d["twg_n"] * MFMA_M >= MIN_DIM)
    grid.filter("ps", "k_factor", check=lambda d: d["k_factor"] > max(PS[d["ps"]].values()))
    grid.filter("twg_k", "k_factor", check=lambda d: d["k_factor"] * d["twg_k"] * 32 >= MIN_DIM)


# Standard pin mapping for CLI args -> axis names.
GEMM_SWEEP_PIN_MAP = {
    "num_workgroups_m": "wg_m_pin",
    "num_workgroups_n": "wg_n_pin",
    "waves_per_wg_m": "waves_m",
    "waves_per_wg_n": "waves_n",
    "tiles_per_wg_m": "twg_m",
    "tiles_per_wg_n": "twg_n",
    "tiles_per_wg_k": "twg_k",
    "k_scaling_factor": "k_factor",
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


# -- Sweep verification -----------------------------------------------------


def verify_top_configs(
    results: list,
    hsaco_paths: dict,
    repro_cmd_fn: Callable,
    top_n: int = 100,
    num_gpus: Optional[int] = None,
    label: str = "",
) -> None:
    """Phase 3: Verify top N configs for correctness using subprocess isolation."""
    if not results:
        return
    if num_gpus is None:
        num_gpus = detect_num_gpus()
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
