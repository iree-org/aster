"""Benchmark: Weak-scaling TFLOPS sweep for Python multi-tile GEMM (test_102).

Single config (repro):
    python .../bench_perf_102_... m4864xn4096xk8192_wg38x32x1_w2x2x1_twg8x8x1_...

Sweep:
    python .../bench_perf_102_... --compile-sample 100
    python .../bench_perf_102_... --tiles-per-wg-m 4 --tiles-per-wg-n 4
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
    MFMA_M,
    SweepGrid,
    add_gemm_sweep_axes,
    add_geometry_pin_args,
    add_resource_filter,
    fits_on_cu_post_compile,
    is_label,
    nwgcu,
    query_gpu_hw,
    resolve_derived_pins,
    verify_top_configs,
    wg_m,
    wg_n,
)


# --- Constants ---

_HW = query_gpu_hw()


# --- Sweep grid ---


def _build_instance(d: dict) -> MultitileGemmInstance:
    _wg_m, _wg_n = wg_m(d, _HW), wg_n(d)
    _nwgcu = nwgcu(d, _HW)
    M = _wg_m * d["twg_m"] * MFMA_M
    N = _wg_n * d["twg_n"] * MFMA_M
    K = d["k_factor"] * d["twg_k"] * 32
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
    return GemmMappingSpec(
        num_workgroups_per_kernel=[wg_m(d, _HW), wg_n(d), 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"]),
        num_wg_per_cu=nwgcu(d, _HW),
        lds_at_write=d["lds_at_write"],
        dealloc_at_read=True,  # test_102 builder deallocates LDS at READ stage
    )


def make_sweep_grid(variants: list[str], check_regs: bool = True) -> SweepGrid:
    grid = SweepGrid()
    grid.axis("variant", variants)
    grid.axis("lds_at_write", [False, True])
    add_gemm_sweep_axes(grid, _HW)
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
    parser.add_argument("--k-scaling-factor", type=int, help="Pin K scaling factor")
    parser.add_argument("--desired-simd-occupancy", type=int, default=None, help="Pin SIMD occupancy")
    parser.add_argument("--direct-b", action=argparse.BooleanOptionalAction, default=None, help="B via preshuffle")
    parser.add_argument("--set-mfma-priority", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

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
    )
    if "_wg_m" in (pins or {}):
        target = pins.pop("_wg_m")
        grid.filter("waves_m", "waves_n", "occ", check=lambda d, t=target: wg_m(d, _HW) == t)

    all_configs, total = grid.generate(
        pins=pins or None,
        sample_size=getattr(args, "compile_sample", 4096),
        stratification_key=lambda d: d["variant"],
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
