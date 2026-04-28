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
    bench_perf_sweep_pipelined,
    make_sweep_pins,
    require_gpu_or_compile_only,
    run_single,
    warn_mcpu_mismatch,
)
from bench_sweep_heuristic import add_heuristic_cli_args, generate_with_weak_scale
from sweep_harness import (
    GEMM_SWEEP_PIN_MAP,
    SweepGrid,
    add_gemm_sweep_axes,
    add_geometry_pin_args,
    add_resource_filter,
    add_size_cli_args,
    apply_wg_pin_filters,
    fits_on_cu_post_compile,
    hw_for_target,
    is_label,
    mapping_kwargs_from_sweep,
    nwgcu,
    parse_size_args,
    resolve_derived_pins,
    verify_top_configs,
)


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
    _wg_m, rem_m = divmod(M, d["twg_m"] * _TILE_ELTS[0])
    _wg_n, rem_n = divmod(N, d["twg_n"] * _TILE_ELTS[1])
    assert rem_m == 0, f"M={M} not divisible by twg_m*tile_m={d['twg_m'] * _TILE_ELTS[0]}"
    assert rem_n == 0, f"N={N} not divisible by twg_n*tile_n={d['twg_n'] * _TILE_ELTS[1]}"
    _nwgcu = nwgcu(d, hw)
    spec = GemmSpec.from_sizes(M, N, K)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"][0]),
        num_wg_per_cu=_nwgcu,
        dealloc_at_read=True,
        mcpu=mcpu,
        **mapping_kwargs_from_sweep(d),
    )
    return PingPongGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict, mcpu: str, hw) -> GemmMappingSpec:
    _wg_m, rem_m = divmod(d["target_M"], d["twg_m"] * _TILE_ELTS[0])
    _wg_n, rem_n = divmod(d["target_N"], d["twg_n"] * _TILE_ELTS[1])
    assert rem_m == 0, f"target_M={d['target_M']} not divisible by twg_m*tile_m={d['twg_m'] * _TILE_ELTS[0]}"
    assert rem_n == 0, f"target_N={d['target_N']} not divisible by twg_n*tile_n={d['twg_n'] * _TILE_ELTS[1]}"
    return GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"][0]),
        num_wg_per_cu=nwgcu(d, hw),
        lds_at_write=d["lds_at_write"],
        dealloc_at_read=True,
        mcpu=mcpu,
    )


def make_sweep_grid(
    variants: list[str],
    mcpu: str,
    hw,
    check_regs: bool = True,
    *,
    target_m: int,
    target_n: int,
    target_k: int,
) -> SweepGrid:
    grid = SweepGrid()
    # Second slot is the load_type; 103 does not sweep it, so it is None.
    grid.axis("variant", [(v, None) for v in variants])
    grid.axis("lds_at_write", [False, True])
    add_gemm_sweep_axes(
        grid,
        hw,
        _TILE_ELTS,
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
    )
    grid.filter("waves_m", "waves_n", check=lambda d: d["waves_m"] * d["waves_n"] == 8)
    grid.axis("set_mfma_priority", [True, False])

    if check_regs:
        add_resource_filter(
            grid,
            hw,
            functools.partial(_mapping_for_resource_check, mcpu=mcpu, hw=hw),
            deps=(
                "variant",
                "target_M",
                "target_N",
                "waves_m",
                "waves_n",
                "occ",
                "twg_m",
                "twg_n",
                "twg_k",
                "ps",
                "lds_at_write",
            ),
        )

    grid.build_with(functools.partial(_build_instance, mcpu=mcpu, hw=hw))
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_103_gemm_python_multitile_ping_pong.py {cfg.label}"


def _from_label(label: str, mcpu: str) -> PingPongGemmInstance:
    base = WeakScaledMappedGemmInstance.from_label(label)
    # Label serde does not encode mcpu; honor --mcpu from the CLI.
    mapping = dataclasses.replace(base.mapping, mcpu=mcpu)
    return PingPongGemmInstance(base.spec, mapping)


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
        run_single(cfg, compile_ping_pong_gemm, args, execute_fn=execute_ping_pong_hsaco)
        return

    parser = argparse.ArgumentParser(description="Python ping-pong GEMM benchmark sweep (test_103)")
    add_sweep_cli_args(parser)
    add_geometry_pin_args(parser)
    add_size_cli_args(parser)
    add_heuristic_cli_args(parser)
    parser.add_argument("--set-mfma-priority", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    warn_mcpu_mismatch(args.mcpu)
    require_gpu_or_compile_only(args)

    hw = hw_for_target(args.mcpu)

    target_m, target_n, target_k = parse_size_args(args, parser)
    print(f"Size: M={target_m}, N={target_n}, K={target_k}  mcpu={args.mcpu}")

    all_paths = ["lds", "direct_b"]
    if args.direct_b is True:
        all_paths = ["direct_b"]
    elif args.direct_b is False:
        all_paths = ["lds"]
    print(f"Variants: {', '.join(all_paths)}")

    pins = make_sweep_pins(args, GEMM_SWEEP_PIN_MAP)
    pins = resolve_derived_pins(pins or {})

    grid = make_sweep_grid(
        all_paths,
        args.mcpu,
        hw,
        check_regs=not getattr(args, "no_reg_filter", False),
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
    )
    apply_wg_pin_filters(grid, pins, _TILE_ELTS[0], _TILE_ELTS[1])

    all_configs, total = generate_with_weak_scale(
        grid,
        args.mcpu,
        "103",
        target_m,
        target_n,
        target_k,
        args,
        sample_size=getattr(args, "compile_sample", 4096),
        pins=pins,
        stratification_key=lambda d: d["variant"],
    )

    results = bench_perf_sweep_pipelined(
        configs=all_configs,
        compile_fn=compile_ping_pong_gemm,
        repro_cmd_fn=_repro_cmd,
        mcpu=args.mcpu,
        num_gpus=0 if args.compile_only else args.num_gpus,
        compile_workers=args.compile_workers,
        compile_timeout=args.compile_timeout,
        post_compile_filter=fits_on_cu_post_compile,
        zero_init=args.zero_init,
        iterations=args.iterations,
    )
    results, hsaco_map = results
    if not args.compile_only:
        verify_top_configs(
            results, hsaco_map, _repro_cmd, mcpu=args.mcpu, top_n=50, num_gpus=args.num_gpus, label="103"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
