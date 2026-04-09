"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (16x16x16 MFMA + dwordx4).

Single config (repro):
    python .../bench_perf_001_... m4864xn4096xk8192_wg38x32_w2x2_twg8x8x1_...

Sweep:
    python .../bench_perf_001_... --use-buffer
    python .../bench_perf_001_... --direct-b --num-gpus 8 --compile-workers 16

If the first positional argument looks like a serialized label, it deserializes
and runs a single config. Otherwise it runs a sweep with the specified parameters.
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
    LoadType,
    OperandPath,
    WeakScaledMappedGemmInstance,
)
from test_perf_001_gemm_fp16_weak_scaled import (
    MLIR_FILES,
    compile_gemm,
    execute_gemm_hsaco,
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
_SPEC = GemmSpec.from_sizes(16, 16, 32)
_TILE_ELTS = GemmMappingSpec(
    num_workgroups_per_kernel=[1, 1, 1],
    num_waves_per_workgroup=[1, 1, 1],
    num_tiles_per_wave=[1, 1, 1],
).tile_elements(_SPEC.mfma_shape)


# --- Sweep grid ---


def _build_instance(d: dict) -> WeakScaledMappedGemmInstance:
    _wg_m, _wg_n = wg_m(d, _HW), wg_n(d)
    M = d["target_M"]
    N = d["target_N"]
    K = d["target_K"]
    spec = GemmSpec.from_sizes(M, N, K)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        load_type=LoadType(d["variant"][1]),
        operand_path=OperandPath(d["variant"][0]),
        num_wg_per_cu=nwgcu(d, _HW),
        lcm_unroll=d["lcm_unroll"],
        unroll_factor_multiplier=d["unroll_mult"],
        epilogue_peeling=d["epilogue_peeling"],
        ll_sched=d["ll_sched"],
        hoist_wait=d["hoist_wait"],
    )
    return WeakScaledMappedGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict) -> GemmMappingSpec:
    return GemmMappingSpec(
        num_workgroups_per_kernel=[wg_m(d, _HW), wg_n(d), 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        load_type=LoadType(d["variant"][1]),
        operand_path=OperandPath(d["variant"][0]),
        num_wg_per_cu=nwgcu(d, _HW),
    )


def make_sweep_grid(variants, check_regs: bool = True) -> SweepGrid:
    grid = SweepGrid()
    grid.axis("variant", [v for v in variants if v in MLIR_FILES])
    add_gemm_sweep_axes(grid, _HW, _TILE_ELTS)

    if check_regs:
        add_resource_filter(
            grid,
            _HW,
            _mapping_for_resource_check,
            deps=("variant", "waves_m", "waves_n", "occ", "twg_m", "twg_n", "twg_k", "ps"),
        )

    grid.build_with(_build_instance)
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_001_gemm_fp16_weak_scaled.py {cfg.label}"


# --- Variant CLI parsing (001-specific: lds/direct/buffer/flat) ---


def _parse_variants(args, parser):
    load_types = []
    if args.use_flat is not False:
        load_types.append("flat")
    if args.use_buffer is not False:
        load_types.append("buffer")
    if args.use_flat is True:
        load_types = [lt for lt in load_types if lt == "flat"]
    if args.use_buffer is True:
        load_types = [lt for lt in load_types if lt == "buffer"]

    if args.direct_a is True and args.direct_b is False:
        parser.error("--direct-a with --no-direct-b is contradictory")
    all_paths = ["lds", "direct_b", "direct_ab"]
    if args.direct_b is False:
        all_paths = ["lds"]
    elif args.direct_b is True and args.direct_a is None:
        all_paths = ["direct_b", "direct_ab"]
    elif args.direct_b is True and args.direct_a is True:
        all_paths = ["direct_ab"]
    elif args.direct_b is True and args.direct_a is False:
        all_paths = ["direct_b"]
    elif args.direct_a is True:
        all_paths = ["direct_ab"]
    elif args.direct_a is False:
        all_paths = ["lds", "direct_b"]

    variants = [(bp, lt) for lt in load_types for bp in all_paths]
    return [(bp, lt) for bp, lt in variants if (bp, lt) in MLIR_FILES]


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        cfg = WeakScaledMappedGemmInstance.from_label(args.label)
        run_single(cfg, compile_gemm, args, execute_fn=execute_gemm_hsaco)
        return

    parser = argparse.ArgumentParser(description="Weak-scaled 16x16+dwordx4 GEMM benchmark sweep")
    add_sweep_cli_args(parser)
    add_geometry_pin_args(parser)
    parser.add_argument("--k-scaling-factor", type=int, help="Pin K scaling factor")
    parser.add_argument("--desired-simd-occupancy", type=int, default=None, help="Pin SIMD occupancy")
    parser.add_argument("--use-buffer", action=argparse.BooleanOptionalAction, default=None, help="Buffer load/store")
    parser.add_argument("--use-flat", action=argparse.BooleanOptionalAction, default=None, help="Flat load/store")
    parser.add_argument("--direct-b", action=argparse.BooleanOptionalAction, default=None, help="B via preshuffle")
    parser.add_argument("--direct-a", action=argparse.BooleanOptionalAction, default=None, help="A via preshuffle")
    args = parser.parse_args()

    variants = _parse_variants(args, parser)
    print(f"Variants: {', '.join(f'{bp}/{lt}' for bp, lt in variants)}")

    pins = make_sweep_pins(args, GEMM_SWEEP_PIN_MAP)
    pins = resolve_derived_pins(pins or {})

    grid = make_sweep_grid(variants, check_regs=not getattr(args, "no_reg_filter", False))
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
        compile_fn=compile_gemm,
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
    verify_top_configs(results, hsaco_map, _repro_cmd, top_n=100, num_gpus=args.num_gpus)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
