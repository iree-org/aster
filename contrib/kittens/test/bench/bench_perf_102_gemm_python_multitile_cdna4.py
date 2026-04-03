"""Benchmark: Weak-scaling TFLOPS sweep for CDNA4 G2S GEMM (test_102_cdna4).

Single config (repro):
    python .../bench_perf_102_...cdna4.py m32xn32xk128_wg1x1x1_w2x2x1_twg2x2x4_pipestrat0_flat_cdna4

Sweep (default M=N=K=4096):
    python .../bench_perf_102_...cdna4.py --compile-sample 100

Pin dimensions:
    python .../bench_perf_102_...cdna4.py --compile-sample 500 --size 2432x12288x4096
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
)
from test_102_gemm_python_multitile_cdna4 import (
    Cdna4GemmInstance,
    compile_cdna4_gemm,
    execute_cdna4_hsaco,
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


# --- Constants ---

# CDNA4 is gfx950-only; ignore whatever hardware the current host reports.
_MCPU = "gfx950"
_HW = query_gpu_hw()
_MFMA_SHAPE = list(Cdna4GemmInstance.MFMA_SHAPE)
_TILE_M, _TILE_N, _TILE_K = GemmMappingSpec.default_tile_elements(_MFMA_SHAPE)


# --- Sweep grid ---


def _build_instance(d: dict) -> Cdna4GemmInstance:
    M, N, K = d["target_M"], d["target_N"], d["target_K"]
    _wg_m, rem_m = divmod(M, d["twg_m"] * _TILE_M)
    _wg_n, rem_n = divmod(N, d["twg_n"] * _TILE_N)
    assert rem_m == 0, f"M={M} not divisible by twg_m*tile_m={d['twg_m'] * _TILE_M}"
    assert rem_n == 0, f"N={N} not divisible by twg_n*tile_n={d['twg_n'] * _TILE_N}"
    _nwgcu = nwgcu(d, _HW)
    spec = GemmSpec.from_sizes(M, N, K, mfma_shape=_MFMA_SHAPE)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=[_wg_m, _wg_n, 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        num_wg_per_cu=_nwgcu,
        lcm_unroll=d["lcm_unroll"],
        unroll_factor_multiplier=d["unroll_mult"],
        epilogue_peeling=d["epilogue_peeling"],
        ll_sched=d["ll_sched"],
        hoist_wait=d["hoist_wait"],
        mcpu=_MCPU,
    )
    return Cdna4GemmInstance(spec, mapping)


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
        num_wg_per_cu=nwgcu(d, _HW),
        mcpu=_MCPU,
    )


def make_sweep_grid(
    check_regs: bool = True,
    *,
    target_m: int,
    target_n: int,
    target_k: int,
) -> SweepGrid:
    grid = SweepGrid()
    add_gemm_sweep_axes(grid, _HW, [_TILE_M, _TILE_N, _TILE_K], target_m=target_m, target_n=target_n, target_k=target_k)

    if check_regs:
        add_resource_filter(
            grid,
            _HW,
            _mapping_for_resource_check,
            deps=("waves_m", "waves_n", "occ", "twg_m", "twg_n", "twg_k", "ps"),
        )

    grid.build_with(_build_instance)
    return grid


# --- Repro ---


def _repro_cmd(cfg):
    return f"python contrib/kittens/test/bench/bench_perf_102_gemm_python_multitile_cdna4.py {cfg.label}"


def _from_label(label: str) -> Cdna4GemmInstance:
    return Cdna4GemmInstance.from_label(label)


# --- Entry point ---


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional and is_label(positional[0]):
        parser = argparse.ArgumentParser(description="Single-config CDNA4 G2S GEMM benchmark (label from sweep)")
        parser.add_argument("label", type=str, help="Config label from sweep output")
        add_single_cli_args(parser)
        args = parser.parse_args()
        cfg = _from_label(args.label)
        run_single(cfg, compile_cdna4_gemm, args, execute_fn=execute_cdna4_hsaco)
        return

    parser = argparse.ArgumentParser(description="CDNA4 G2S GEMM benchmark sweep (test_102_cdna4)")
    add_sweep_cli_args(parser)
    add_geometry_pin_args(parser)
    add_size_cli_args(parser)
    args = parser.parse_args()

    target_m, target_n, target_k = parse_size_args(args, parser)
    print(f"Size: M={target_m}, N={target_n}, K={target_k}")

    pins = make_sweep_pins(args, GEMM_SWEEP_PIN_MAP)
    pins = resolve_derived_pins(pins or {})

    grid = make_sweep_grid(
        check_regs=not getattr(args, "no_reg_filter", False),
        target_m=target_m,
        target_n=target_n,
        target_k=target_k,
    )
    apply_wg_pin_filters(grid, pins, _TILE_M, _TILE_N)

    all_configs, total = grid.generate(
        pins=pins or None,
        sample_size=getattr(args, "compile_sample", 4096),
        stratification_key=lambda d: (d["waves_m"], d["waves_n"]),
    )

    results = bench_perf_sweep_pipelined(
        configs=all_configs,
        compile_fn=compile_cdna4_gemm,
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
    verify_top_configs(results, hsaco_map, _repro_cmd, top_n=50, num_gpus=args.num_gpus, label="102_cdna4")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
