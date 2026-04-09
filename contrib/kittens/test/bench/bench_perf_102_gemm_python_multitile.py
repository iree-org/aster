"""Benchmark: Weak-scaling TFLOPS sweep for Python multi-tile GEMM (test_102).

Single config (repro):
    python .../bench_perf_102_... m4864xn4096xk8192_wg38x32x1_w2x2x1_twg8x8x1_...

Sweep (default M=N=K=4096):
    python .../bench_perf_102_... --compile-sample 100

Pin individual dimensions (others default to 4096):
    python .../bench_perf_102_... --compile-sample 500 --m 2432 --k 128

Pin all three at once (exclusive with --m/--n/--k):
    python .../bench_perf_102_... --compile-sample 500 --size 2432x12288x4096

Heuristic-guided sweep:
    python .../bench_perf_102_... --compile-sample 500 --heuristic
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

import argparse
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
from test_102_gemm_python_multitile import (
    MultitileGemmInstance,
    compile_multitile_gemm,
    execute_multitile_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    bench_perf_sweep_pipelined,
    make_sweep_pins,
    run_single,
)
from sweep_harness import (
    GEMM_SWEEP_PIN_MAP,
    SweepGrid,
    add_gemm_sweep_axes,
    add_geometry_pin_args,
    add_resource_filter,
    add_size_cli_args,
    apply_wg_pin_filters,
    fits_on_cu_post_compile,
    is_label,
    nwgcu,
    parse_size_args,
    query_gpu_hw,
    resolve_derived_pins,
    verify_top_configs,
)
from bench_sweep_heuristic import add_heuristic_cli_args, make_score_fn


# --- Constants ---

_HW = query_gpu_hw()
# Default spec + mapping for tile_elements derivation.
_SPEC = GemmSpec.from_sizes(16, 16, 32)
_MAPPING = GemmMappingSpec(
    num_workgroups_per_kernel=[1, 1, 1],
    num_waves_per_workgroup=[1, 1, 1],
    num_tiles_per_wave=[1, 1, 1],
)
_TILE_M, _TILE_N, _TILE_K = _MAPPING.tile_elements(_SPEC.mfma_shape)


# --- Sweep grid ---


def _build_instance(d: dict) -> MultitileGemmInstance:
    M, N, K = d["target_M"], d["target_N"], d["target_K"]
    _wg_m, rem_m = divmod(M, d["twg_m"] * _TILE_M)
    _wg_n, rem_n = divmod(N, d["twg_n"] * _TILE_N)
    assert rem_m == 0, f"M={M} not divisible by twg_m*tile_m={d['twg_m'] * _TILE_M}"
    assert rem_n == 0, f"N={N} not divisible by twg_n*tile_n={d['twg_n'] * _TILE_N}"
    _nwgcu = nwgcu(d, _HW)
    spec = GemmSpec.from_sizes(M, N, K)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"]),
        num_wg_per_cu=_nwgcu,
        lcm_unroll=d["lcm_unroll"],
        unroll_factor_multiplier=d["unroll_mult"],
        epilogue_peeling=d["epilogue_peeling"],
        ll_sched=d["ll_sched"],
        hoist_wait=d["hoist_wait"],
        lds_at_write=d["lds_at_write"],
        dealloc_at_read=True,
        set_mfma_priority=d["set_mfma_priority"],
    )
    return MultitileGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict) -> GemmMappingSpec:
    _wg_m, rem_m = divmod(d["target_M"], d["twg_m"] * _TILE_M)
    _wg_n, rem_n = divmod(d["target_N"], d["twg_n"] * _TILE_N)
    assert rem_m == 0, f"target_M={d['target_M']} not divisible by twg_m*tile_m={d['twg_m'] * _TILE_M}"
    assert rem_n == 0, f"target_N={d['target_N']} not divisible by twg_n*tile_n={d['twg_n'] * _TILE_N}"
    return GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"]),
        num_wg_per_cu=nwgcu(d, _HW),
        lds_at_write=d["lds_at_write"],
        dealloc_at_read=True,
    )


def make_sweep_grid(
    variants: list[str],
    check_regs: bool = True,
    *,
    target_m: int,
    target_n: int,
    target_k: int,
) -> SweepGrid:
    grid = SweepGrid()
    grid.axis("variant", variants)
    grid.axis("lds_at_write", [False, True])
    add_gemm_sweep_axes(grid, _HW, [_TILE_M, _TILE_N, _TILE_K], target_m=target_m, target_n=target_n, target_k=target_k)
    grid.axis("set_mfma_priority", [True, False])

    if check_regs:
        add_resource_filter(
            grid,
            _HW,
            _mapping_for_resource_check,
            deps=("variant", "waves_m", "waves_n", "occ", "twg_m", "twg_n", "twg_k", "ps", "lds_at_write"),
        )

    grid.build_with(_build_instance)
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_102_gemm_python_multitile.py {cfg.label}"


def _from_label(label: str) -> MultitileGemmInstance:
    base = WeakScaledMappedGemmInstance.from_label(label)
    return MultitileGemmInstance(base.spec, base.mapping)


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config multitile GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        cfg = _from_label(args.label)
        run_single(cfg, compile_multitile_gemm, args, execute_fn=execute_multitile_hsaco)
        return

    parser = argparse.ArgumentParser(description="Python multi-tile GEMM benchmark sweep (test_102)")
    add_sweep_cli_args(parser)
    add_geometry_pin_args(parser)
    add_size_cli_args(parser)
    add_heuristic_cli_args(parser)
    parser.add_argument("--set-mfma-priority", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    target_m, target_n, target_k = parse_size_args(args, parser)
    print(f"Size: M={target_m}, N={target_n}, K={target_k}")

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
        check_regs=not getattr(args, "no_reg_filter", False),
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
    )
    apply_wg_pin_filters(grid, pins, _TILE_M, _TILE_N)

    priority_fn = make_score_fn("102") if args.heuristic else None
    all_configs, total = grid.generate(
        pins=pins or None,
        sample_size=getattr(args, "compile_sample", 4096),
        stratification_key=None if priority_fn else (lambda d: d["variant"]),
        priority_fn=priority_fn,
    )

    results = bench_perf_sweep_pipelined(
        configs=all_configs,
        compile_fn=compile_multitile_gemm,
        repro_cmd_fn=_repro_cmd,
        num_gpus=args.num_gpus,
        compile_workers=args.compile_workers,
        compile_timeout=args.compile_timeout,
        post_compile_filter=fits_on_cu_post_compile,
        exec_sample=getattr(args, "exec_sample", 2000),
        zero_init=args.zero_init,
        iterations=args.iterations,
    )
    results, hsaco_map = results
    verify_top_configs(results, hsaco_map, _repro_cmd, top_n=50, num_gpus=args.num_gpus, label="102")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
