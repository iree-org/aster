"""Benchmark: Weak-scaling TFLOPS sweep for Python multi-tile GEMM (test_102, lds, CDNA3).

This leaf is the LDS-only specialization of bench_perf_102_gemm_python_multitile.
Operand path is hardcoded to OperandPath.LDS; no path axis is swept.

Single config (repro):
    python .../bench_perf_102_..._lds_cdna3 m4864xn4096xk8192_wg38x32x1_w2x2x1_twg8x8x1_...

Sweep (default M=N=K=4096):
    python .../bench_perf_102_..._lds_cdna3 --compile-sample 100

Pin individual dimensions (others default to 4096):
    python .../bench_perf_102_..._lds_cdna3 --compile-sample 500 --m 2432 --k 128

Pin all three at once (exclusive with --m/--n/--k):
    python .../bench_perf_102_..._lds_cdna3 --compile-sample 500 --size 2432x12288x4096

Heuristic-guided sweep:
    python .../bench_perf_102_..._lds_cdna3 --compile-sample 500 --heuristic
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

import argparse
import dataclasses
import functools
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from kittens.gemm_config import (
    GemmSpec,
    GemmMappingSpec,
    OperandPath,
    WeakScaledMappedGemmInstance,
    mfma_shape_for_mcpu,
)
from test_102_gemm_python_multitile_lds_cdna3 import (
    MultitileGemmInstance,
    compile_multitile_gemm,
    execute_multitile_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    require_gpu_or_compile_only,
    run_single,
    warn_mcpu_mismatch,
)
from kittens_helpers import PIPELINE_STRATEGIES as PS
from bench_search import (
    SweepGrid,
    add_gemm_sweep_axes,
    add_resource_filter,
    add_size_cli_args,
    apply_bench_scheduling_defaults,
    fits_on_cu_post_compile,
    hw_for_target,
    is_label,
    mapping_kwargs_from_sweep,
    nwgcu,
    parse_size_args,
    wps,
)
from bench_tier_driver import run_tier_mode
from bench_tier_schedule import TierSpec, make_constraints


# --- Tier schedule (policy) ---


def make_tiered_schedule(max_configs: int, random_seed: int, constraints: tuple[str, ...]) -> list[TierSpec]:
    return [
        TierSpec(
            tier_idx=1,
            per_stratum_diversity=True,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                wg_m=[16, 19, 32, 38, 76],
                wg_n=[16, 19, 32, 38, 76],
                waves_m=[1, 2, 4],
                waves_n=[1, 2, 4],
                twg_m=[4, 6, 8, 12, 16],
                twg_n=[4, 6, 8, 12, 16],
                twg_k=[1, 2, 4],
                occ=[1, 2],
                ps=[1, 3, 5],
                unroll_factor_multiplier=[1, 3],
                ll_sched=[1],
                rotate_compute_stage=[True],
                hoist_wait=[False],
                use_conservative_barriers=[False, True],
            ),
            discriminator=("wg_m", "wg_n"),
        ),
        TierSpec(
            tier_idx=2,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                ll_sched=[0, 1, 2, 3, 4, 5],
                rotate_compute_stage=[True, False],
                hoist_wait=[True, False],
                epilogue_peeling=[True, False],
                use_conservative_barriers=[False, True],
            ),
            anchor_axes=dict(ll_sched=1, rotate_compute_stage=True),
            discriminator=("hoist_wait", "ll_sched", "rotate_compute_stage", "use_conservative_barriers"),
        ),
        TierSpec(
            tier_idx=3,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                ps=[1, 2, 3, 4, 5, 6, 7, 8],
                unroll_factor_multiplier=[1, 2, 3, 4, 5],
                ll_sched=[0, 1, 2, 3, 4, 5],
                rotate_compute_stage=[True, False],
                hoist_wait=[True, False],
                epilogue_peeling=[True, False],
                use_conservative_barriers=[False, True],
            ),
            anchor_axes=dict(ll_sched=1, rotate_compute_stage=True),
            discriminator=("ps", "unroll_factor_multiplier", "hoist_wait"),
        ),
    ]


# --- Constants ---


def _tile_elements(mcpu: str) -> tuple[int, int, int]:
    """Tile elements [M, N, K] for the given target's MFMA shape."""
    mfma = mfma_shape_for_mcpu(mcpu)
    return tuple(GemmMappingSpec.default_tile_elements(mfma))


# --- Sweep grid ---


def _build_instance(d: dict, mcpu: str, hw) -> MultitileGemmInstance:
    M, N, K = d["target_M"], d["target_N"], d["target_K"]
    tile_m, tile_n, _ = _tile_elements(mcpu)
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    _nwgcu = nwgcu(d, hw)
    mfma = mfma_shape_for_mcpu(mcpu)
    spec = GemmSpec.from_sizes(M, N, K, mfma_shape=mfma)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.LDS,
        num_wg_per_cu=_nwgcu,
        mcpu=mcpu,
        **mapping_kwargs_from_sweep(d),
    )
    return MultitileGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict, mcpu: str, hw) -> GemmMappingSpec:
    tile_m, tile_n, _ = _tile_elements(mcpu)
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    return GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.LDS,
        num_wg_per_cu=nwgcu(d, hw),
        mcpu=mcpu,
    )


def _make_grid(
    *,
    mcpu: str,
    hw,
    target_m: int,
    target_n: int,
    target_k: int,
    check_regs: bool = True,
) -> SweepGrid:
    """Return a fresh SweepGrid populated with bench_perf_102's axes + filters + builder."""
    tile_m, tile_n, tile_k = _tile_elements(mcpu)
    grid = SweepGrid()
    add_gemm_sweep_axes(grid)
    apply_bench_scheduling_defaults(grid)

    grid.filter(
        "lcm_unroll",
        "unroll_factor_multiplier",
        check=lambda d: d["lcm_unroll"] or d["unroll_factor_multiplier"] == 1,
        name="unroll_multiplier_gt_1_implies_lcm_unroll",
    )
    grid.filter(
        "waves_m",
        "waves_n",
        check=lambda d, ns=hw.num_simds: ns <= d["waves_m"] * d["waves_n"] <= 4 * ns
        and (d["waves_m"] * d["waves_n"]) % ns == 0,
        name="wave_pair_valid",
    )
    grid.filter(
        "waves_m",
        "waves_n",
        "occ",
        check=lambda d, h=hw: d["occ"] % wps(d, h) == 0,
        name="occ_divisible_by_waves_per_simd",
    )
    grid.filter(
        "waves_m",
        "twg_m",
        check=lambda d: d["twg_m"] % d["waves_m"] == 0,
        name="twg_m_divisible_by_waves_m",
    )
    grid.filter(
        "waves_n",
        "twg_n",
        check=lambda d: d["twg_n"] % d["waves_n"] == 0,
        name="twg_n_divisible_by_waves_n",
    )
    grid.filter(
        "target_M",
        "wg_m",
        "twg_m",
        check=lambda d, t=tile_m: d["target_M"] == d["wg_m"] * d["twg_m"] * t,
        name="target_M_divisible_by_twg_m_times_tile_m",
    )
    grid.filter(
        "target_N",
        "wg_n",
        "twg_n",
        check=lambda d, t=tile_n: d["target_N"] == d["wg_n"] * d["twg_n"] * t,
        name="target_N_divisible_by_twg_n_times_tile_n",
    )
    grid.filter(
        "target_K",
        "twg_k",
        check=lambda d, t=tile_k: d["target_K"] % (d["twg_k"] * t) == 0,
        name="target_K_divisible_by_twg_k_times_tile_k",
    )
    _ps_max = {ps_id: max(stages.values()) for ps_id, stages in PS.items()}
    grid.filter(
        "target_K",
        "twg_k",
        "ps",
        check=lambda d, t=tile_k, pm=_ps_max: d["target_K"] // (d["twg_k"] * t) > pm[d["ps"]],
        name="k_iters_exceed_pipeline_max_stage",
    )
    grid.filter(
        "twg_m",
        "twg_n",
        "twg_k",
        check=lambda d: (d["twg_m"] * d["twg_n"] * d["twg_k"]) >= 48 and (d["twg_m"] * d["twg_n"] * d["twg_k"]) <= 256,
        name="num_tiles_constrainted_in_48_256",
    )

    if check_regs:
        add_resource_filter(
            grid,
            hw,
            functools.partial(_mapping_for_resource_check, mcpu=mcpu, hw=hw),
            deps=(
                "waves_m",
                "waves_n",
                "occ",
                "twg_m",
                "twg_n",
                "twg_k",
                "ps",
            ),
        )

    grid.build_with(functools.partial(_build_instance, mcpu=mcpu, hw=hw))
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_102_gemm_python_multitile_lds_cdna3.py {cfg.label}"


def _from_label(label: str, mcpu: str) -> MultitileGemmInstance:
    base = WeakScaledMappedGemmInstance.from_label(label)
    mfma = mfma_shape_for_mcpu(mcpu)
    spec = dataclasses.replace(base.spec, mfma_shape=mfma)
    mapping = dataclasses.replace(base.mapping, mcpu=mcpu, operand_path=OperandPath.LDS)
    return MultitileGemmInstance(spec, mapping)


# Hooks consumed by perf_evaluate so it can drive bench_perf_sweep_pipelined directly.
BENCH_HOOKS = {
    "bench_label": "bench_perf_102_gemm_python_multitile_lds_cdna3",
    "from_label": _from_label,
    "compile_fn": compile_multitile_gemm,
    "repro_cmd_fn": _repro_cmd,
    "post_compile_filter": fits_on_cu_post_compile,
}


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config multitile GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        warn_mcpu_mismatch(args.mcpu)
        require_gpu_or_compile_only(args)
        cfg = _from_label(args.label, args.mcpu)
        run_single(
            cfg,
            compile_multitile_gemm,
            args,
            execute_fn=execute_multitile_hsaco,
            bench="bench_perf_102_gemm_python_multitile_lds_cdna3",
        )
        return

    parser = argparse.ArgumentParser(description="Python multi-tile GEMM benchmark sweep (test_102, lds variant)")
    add_sweep_cli_args(parser)
    add_size_cli_args(parser)
    args = parser.parse_args()
    warn_mcpu_mismatch(args.mcpu)
    require_gpu_or_compile_only(args)

    hw = hw_for_target(args.mcpu)

    target_m, target_n, target_k = parse_size_args(args, parser)
    print(f"Size: M={target_m}, N={target_n}, K={target_k}  mcpu={args.mcpu}")

    grid_factory = functools.partial(
        _make_grid,
        mcpu=args.mcpu,
        hw=hw,
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
        check_regs=not getattr(args, "no_reg_filter", False),
    )

    run_tier_mode(
        args,
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
        grid_factory=grid_factory,
        compile_fn=compile_multitile_gemm,
        repro_cmd_fn=_repro_cmd,
        post_compile_filter=fits_on_cu_post_compile,
        bench_label="bench_perf_102_gemm_python_multitile_lds_cdna3",
        tier_schedule=make_tiered_schedule(args.compile_sample, args.seed, make_constraints(grid_factory())),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
