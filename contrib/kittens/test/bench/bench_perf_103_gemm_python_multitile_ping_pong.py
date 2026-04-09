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


def _build_instance(d: dict) -> PingPongGemmInstance:
    _wg_m, _wg_n = wg_m(d, _HW), wg_n(d)
    _nwgcu = nwgcu(d, _HW)
    M = d["target_M"]
    N = d["target_N"]
    K = d["target_K"]
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
    return PingPongGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict) -> GemmMappingSpec:
    return GemmMappingSpec(
        num_workgroups_per_kernel=[wg_m(d, _HW), wg_n(d), 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        operand_path=OperandPath(d["variant"]),
        num_wg_per_cu=nwgcu(d, _HW),
        lds_at_write=d["lds_at_write"],
        dealloc_at_read=True,
    )


def make_sweep_grid(variants: list[str], check_regs: bool = True) -> SweepGrid:
    grid = SweepGrid()
    grid.axis("variant", variants)
    grid.axis("lds_at_write", [False, True])
    # Sweep M and N freely: wg_m = WG_BASE[0] * nwgcu is a multiple of 19,
    # which does not divide DEFAULT_DIM (4096), so pinning to DEFAULT_DIM
    # would yield an empty grid.  K can stay fixed.
    add_gemm_sweep_axes(grid, _HW, _TILE_ELTS, target_m=None, target_n=None)
    grid.filter("waves_m", "waves_n", check=lambda d: d["waves_m"] * d["waves_n"] == 8)

    # Weak-scale constraint: problem sizes must exactly equal wg * twg * mfma.
    # The divisibility filter in add_gemm_sweep_axes is necessary but not
    # sufficient; these equality filters enforce the exact match required by
    # WeakScaledMappedGemmInstance._check_weak_scale().
    grid.filter(
        "target_M",
        "waves_m",
        "waves_n",
        "occ",
        "twg_m",
        check=lambda d, t=_TILE_ELTS[0]: wg_m(d, _HW) * d["twg_m"] * t == d["target_M"],
    )
    grid.filter(
        "target_N",
        "n_mult",
        "twg_n",
        check=lambda d, t=_TILE_ELTS[1]: wg_n(d) * d["twg_n"] * t == d["target_N"],
    )

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
    return f"python contrib/kittens/test/bench/bench_perf_103_gemm_python_multitile_ping_pong.py {cfg.label}"


def _from_label(label: str) -> PingPongGemmInstance:
    base = WeakScaledMappedGemmInstance.from_label(label)
    return PingPongGemmInstance(base.spec, base.mapping)


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config ping-pong GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        cfg = _from_label(args.label)
        run_single(cfg, compile_ping_pong_gemm, args, execute_fn=execute_ping_pong_hsaco)
        return

    parser = argparse.ArgumentParser(description="Python ping-pong GEMM benchmark sweep (test_103)")
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
        compile_fn=compile_ping_pong_gemm,
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
    verify_top_configs(results, hsaco_map, _repro_cmd, top_n=50, num_gpus=args.num_gpus, label="103")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
