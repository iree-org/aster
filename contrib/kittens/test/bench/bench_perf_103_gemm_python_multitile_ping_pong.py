"""Benchmark: Weak-scaling TFLOPS sweep for Python ping-pong GEMM (test_103).

Single config (repro):
    python .../bench_perf_103_... m4864xn4096xk8192_wg38x32x1_w2x2x1_twg8x8x1_...

Sweep:
    python .../bench_perf_103_... --compile-sample 100
    python .../bench_perf_103_... --tiles-per-wg-m 8 --tiles-per-wg-n 8
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
)
from test_103_gemm_python_multitile_ping_pong import (
    PingPongGemmInstance,
    compile_ping_pong_gemm,
    execute_ping_pong_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    require_gpu_or_compile_only,
    run_single,
    warn_mcpu_mismatch,
)
from bench_tier_driver import run_tier_mode
from bench_tier_schedule import TierSpec, make_constraints
from kittens_helpers import PIPELINE_STRATEGIES as PS
from bench_search import (
    SweepGrid,
    add_gemm_sweep_axes,
    add_resource_filter,
    add_size_cli_args,
    fits_on_cu_post_compile,
    hw_for_target,
    is_label,
    mapping_kwargs_from_sweep,
    nwgcu,
    parse_size_args,
    wps,
)


# --- Tier schedule (policy) ---


def make_tiered_schedule(max_configs: int, random_seed: int, constraints: tuple[str, ...]) -> list[TierSpec]:
    return [
        TierSpec(
            tier_idx=1,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                wg_m=[8, 16, 19, 32, 38, 64, 76],
                wg_n=[8, 16, 19, 32, 38, 64, 76],
                waves_m=[1, 2, 4],
                waves_n=[1, 2, 4, 8],
                twg_m=[1, 4, 6, 8, 16],
                twg_n=[1, 4, 6, 8, 16],
                twg_k=[1, 2, 4],
                occ=[1, 2, 3],
                ps=[1, 3, 5, 7, 9],
                unroll_factor_multiplier=[1, 2, 3],
                ll_sched=[True, False],
                rotate_compute_stage=[True, False],
                hoist_wait=[True, False],
            ),
            fixed_axes=dict(),
            discriminator=("wg_m", "wg_n"),
        ),
        TierSpec(
            tier_idx=2,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                ll_sched=[True, False],
                rotate_compute_stage=[True, False],
                hoist_wait=[True, False],
                epilogue_peeling=[True, False],
            ),
            anchor_axes=dict(ll_sched=True, rotate_compute_stage=True),
            discriminator=("hoist_wait", "ll_sched", "rotate_compute_stage"),
        ),
        TierSpec(
            tier_idx=3,
            max_configs=max_configs,
            random_seed=random_seed,
            constraints=constraints,
            axis_grid=dict(
                ll_sched=[True, False],
                rotate_compute_stage=[True, False],
                hoist_wait=[True, False],
                epilogue_peeling=[True, False],
            ),
            anchor_axes=dict(ll_sched=True, rotate_compute_stage=True),
            neighbor_radius=dict(
                wg_m=1,
                wg_n=1,
                waves_m=1,
                waves_n=1,
                twg_m=1,
                twg_n=1,
                twg_k=1,
                ps=1,
                unroll_factor_multiplier=1,
            ),
            discriminator="ps",
        ),
    ]


# --- Constants ---

_SPEC = GemmSpec.from_sizes(16, 16, 32)
_TILE_ELTS = GemmMappingSpec(
    num_workgroups_per_kernel=[1, 1, 1],
    num_waves_per_workgroup=[1, 1, 1],
    num_tiles_per_wave=[1, 1, 1],
).tile_elements(_SPEC.mfma_shape)


# --- Sweep grid ---


def _build_instance(d: dict, mcpu: str, hw) -> PingPongGemmInstance:
    M, N, K = d["target_M"], d["target_N"], d["target_K"]
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    _nwgcu = nwgcu(d, hw)
    spec = GemmSpec.from_sizes(M, N, K)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.LDS,
        num_wg_per_cu=_nwgcu,
        dealloc_at_read=d["dealloc_at_read"],
        mcpu=mcpu,
        **mapping_kwargs_from_sweep(d),
    )
    return PingPongGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict, mcpu: str, hw) -> GemmMappingSpec:
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    return GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.LDS,
        num_wg_per_cu=nwgcu(d, hw),
        dealloc_at_read=d["dealloc_at_read"],
        mcpu=mcpu,
    )


def _make_grid(
    *,
    mcpu: str,
    hw,
    check_regs: bool = True,
    target_m: int,
    target_n: int,
    target_k: int,
) -> SweepGrid:
    """Return a fresh SweepGrid populated with this bench's axes + filters + builder."""
    tile_m, tile_n, tile_k = _TILE_ELTS
    grid = SweepGrid()
    grid.axis("dealloc_at_read", [True])
    add_gemm_sweep_axes(grid)
    grid.restrict_axes(
        {
            "lcm_unroll": [True],
            "epilogue_peeling": [False, True],
        }
    )

    grid.filter(
        "lcm_unroll",
        "unroll_factor_multiplier",
        check=lambda d: d["lcm_unroll"] or d["unroll_factor_multiplier"] == 1,
        name="unroll_multiplier_gt_1_implies_lcm_unroll",
    )
    # Ping-pong requires exactly 2 waves per SIMD (4 wave pairs total per workgroup).
    grid.filter(
        "waves_m",
        "waves_n",
        check=lambda d, ns=hw.num_simds: d["waves_m"] * d["waves_n"] == 2 * ns,
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
                "target_M",
                "target_N",
                "waves_m",
                "waves_n",
                "occ",
                "twg_m",
                "twg_n",
                "twg_k",
                "ps",
                "dealloc_at_read",
            ),
        )

    grid.build_with(functools.partial(_build_instance, mcpu=mcpu, hw=hw))
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_103_gemm_python_multitile_ping_pong.py {cfg.label}"


def _from_label(label: str, mcpu: str) -> PingPongGemmInstance:
    base = WeakScaledMappedGemmInstance.from_label(label)
    mapping = dataclasses.replace(base.mapping, mcpu=mcpu, operand_path=OperandPath.LDS)
    return PingPongGemmInstance(base.spec, mapping)


# Hooks consumed by perf_evaluate so it can drive bench_perf_sweep_pipelined directly.
BENCH_HOOKS = {
    "bench_label": "bench_perf_103_gemm_python_multitile_ping_pong",
    "from_label": _from_label,
    "compile_fn": compile_ping_pong_gemm,
    "repro_cmd_fn": _repro_cmd,
    "post_compile_filter": fits_on_cu_post_compile,
}


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config ping-pong GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        warn_mcpu_mismatch(args.mcpu)
        require_gpu_or_compile_only(args)
        cfg = _from_label(args.label, args.mcpu)
        run_single(
            cfg,
            compile_ping_pong_gemm,
            args,
            execute_fn=execute_ping_pong_hsaco,
            bench="bench_perf_103_gemm_python_multitile_ping_pong",
        )
        return

    parser = argparse.ArgumentParser(description="Python ping-pong GEMM benchmark sweep (test_103)")
    add_sweep_cli_args(parser)
    add_size_cli_args(parser)
    parser.add_argument("--set-mfma-priority", action=argparse.BooleanOptionalAction, default=None)
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
        compile_fn=compile_ping_pong_gemm,
        repro_cmd_fn=_repro_cmd,
        post_compile_filter=fits_on_cu_post_compile,
        bench_label="bench_perf_103_gemm_python_multitile_ping_pong",
        tier_schedule=make_tiered_schedule(args.compile_sample, args.seed, make_constraints(grid_factory())),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
