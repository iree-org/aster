"""Benchmark: Weak-scaling TFLOPS sweep for CDNA4 G2S GEMM (test_102_cdna4_directb).

Single config (repro):
    python .../bench_perf_102_..._directb_cdna4.py m32xn32xk128_wg1x1x1_w2x2x1_twg2x2x4_pipestrat0_flat_cdna4

Sweep (default M=N=K=4096):
    python .../bench_perf_102_..._directb_cdna4.py --compile-sample 100

Pin dimensions:
    python .../bench_perf_102_..._directb_cdna4.py --compile-sample 500 --size 2432x12288x4096
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

from aster.dialects.kernel_builder import MFMA_F16_CDNA4
from kittens.gemm_config import (
    GemmSpec,
    GemmMappingSpec,
    OperandPath,
)
from test_102_gemm_python_multitile_g2s_directb_cdna4 import (
    Cdna4GemmInstance,
    compile_cdna4_gemm,
    execute_cdna4_hsaco,
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
                wg_m=[16, 19, 32, 38, 76],
                wg_n=[16, 19, 32, 38, 76],
                waves_m=[1, 2, 4],
                waves_n=[1, 2, 4],
                twg_m=[4, 6, 8, 12, 16],
                twg_n=[4, 6, 8, 12, 16],
                twg_k=[1, 2, 4],
                occ=[1, 2],
                ps=[1, 3, 5, 11, 13],
                unroll_factor_multiplier=[1, 3],
                ll_sched=[True],
                rotate_compute_stage=[True],
                hoist_wait=[False],
            ),
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

_MFMA_SHAPE = list(MFMA_F16_CDNA4.shape)
_TILE_M, _TILE_N, _TILE_K = GemmMappingSpec.default_tile_elements(_MFMA_SHAPE)


# --- Sweep grid ---


def _build_instance(d: dict, mcpu: str, hw) -> Cdna4GemmInstance:
    M, N, K = d["target_M"], d["target_N"], d["target_K"]
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    _nwgcu = nwgcu(d, hw)
    spec = GemmSpec.from_sizes(M, N, K, mfma_shape=_MFMA_SHAPE)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.DIRECT_B,
        num_wg_per_cu=nwgcu(d, hw),
        dealloc_at_read=d["dealloc_at_read"],
        mcpu=mcpu,
        **mapping_kwargs_from_sweep(d),
    )
    return Cdna4GemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict, mcpu: str, hw) -> GemmMappingSpec:
    _wg_m, _wg_n = d["wg_m"], d["wg_n"]
    return GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath.DIRECT_B,
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
    tile_m, tile_n, tile_k = _TILE_M, _TILE_N, _TILE_K
    grid = SweepGrid()
    grid.axis("dealloc_at_read", [True])
    add_gemm_sweep_axes(grid)
    grid.restrict_axes(
        {
            "occ": [1],
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
        check=lambda d: (d["twg_m"] * d["twg_n"] * d["twg_k"]) >= 32 and (d["twg_m"] * d["twg_n"] * d["twg_k"]) <= 256,
        name="num_tiles_constrainted_in_32_256",
    )
    # Total wg count must be a multiple of CU count (avoids tail effects).
    grid.filter(
        "wg_m",
        "wg_n",
        check=lambda d, ncus=hw.num_cus: (d["wg_m"] * d["wg_n"]) % ncus == 0,
        name="wg_count_multiple_of_cus",
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
                "dealloc_at_read",
            ),
        )

    grid.build_with(functools.partial(_build_instance, mcpu=mcpu, hw=hw))
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_102_gemm_python_multitile_g2s_directb_cdna4.py {cfg.label}"


def _from_label(label: str, mcpu: str) -> Cdna4GemmInstance:
    """Accept the bench's `..._cdna4` label or a base CDNA3-format
    `..._b_<path>_lt_<load>` label."""
    if label.endswith("_cdna4"):
        label = label[: -len("_cdna4")]
    cfg = Cdna4GemmInstance.from_label(label)
    cfg.mapping = dataclasses.replace(cfg.mapping, mcpu=mcpu, operand_path=OperandPath.DIRECT_B)
    return cfg


# Hooks consumed by perf_evaluate so it can drive bench_perf_sweep_pipelined directly.
BENCH_HOOKS = {
    "bench_label": "bench_perf_102_gemm_python_multitile_g2s_directb_cdna4",
    "from_label": _from_label,
    "compile_fn": compile_cdna4_gemm,
    "repro_cmd_fn": _repro_cmd,
    "post_compile_filter": fits_on_cu_post_compile,
}


# --- Entry point ---


_CDNA4_MCPU = "gfx950"


def _require_cdna4_mcpu(parser: argparse.ArgumentParser, mcpu: str) -> None:
    """Reject non-CDNA4 mcpu fast: this kernel uses ISA that does not encode on gfx94x."""
    if mcpu != _CDNA4_MCPU:
        parser.error(
            f"--mcpu={mcpu} is not supported by this CDNA4 kernel "
            f"(use --mcpu={_CDNA4_MCPU}; for CDNA3 / gfx94x labels run "
            f"bench_perf_102_gemm_python_multitile.py instead)."
        )


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config CDNA4 G2S GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        _require_cdna4_mcpu(parser, args.mcpu)
        warn_mcpu_mismatch(args.mcpu)
        require_gpu_or_compile_only(args)
        cfg = _from_label(args.label, args.mcpu)
        run_single(
            cfg,
            compile_cdna4_gemm,
            args,
            execute_fn=execute_cdna4_hsaco,
            bench="bench_perf_102_gemm_python_multitile_g2s_directb_cdna4",
        )
        return

    parser = argparse.ArgumentParser(description="CDNA4 G2S GEMM benchmark sweep (test_102_cdna4_directb)")
    add_sweep_cli_args(parser)
    add_size_cli_args(parser)
    args = parser.parse_args()
    _require_cdna4_mcpu(parser, args.mcpu)
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
        compile_fn=compile_cdna4_gemm,
        repro_cmd_fn=_repro_cmd,
        post_compile_filter=fits_on_cu_post_compile,
        bench_label="bench_perf_102_gemm_python_multitile_g2s_directb_cdna4",
        tier_schedule=make_tiered_schedule(args.compile_sample, args.seed, make_constraints(grid_factory())),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
