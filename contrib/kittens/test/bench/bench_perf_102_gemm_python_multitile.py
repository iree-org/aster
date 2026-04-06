# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Benchmark: Weak-scaling TFLOPS sweep for Python multi-tile GEMM (test_102).

Single config (repro):
    python .../bench_perf_102_... m4864xn4096xk8192_wg38x32x1_w2x2x1_twg8x8x1_...

Sweep:
    python .../bench_perf_102_... --compile-sample 100
    python .../bench_perf_102_... --tiles-per-wg-m 4 --tiles-per-wg-n 4

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
        num_wg_per_cu=_nwgcu,
        lcm_unroll=d["lcm_unroll"],
        unroll_factor_multiplier=d["unroll_mult"],
        epilogue_peeling=d["epilogue_peeling"],
        ll_sched=d["ll_sched"],
        hoist_wait=d["hoist_wait"],
    )
    return MultitileGemmInstance(spec, mapping)


def _mapping_for_resource_check(d: dict) -> GemmMappingSpec:
    return GemmMappingSpec(
        num_workgroups_per_kernel=[wg_m(d, _HW), wg_n(d), 1],
        num_waves_per_workgroup=[d["waves_m"], d["waves_n"], 1],
        num_tiles_per_wave=[d["twg_m"] // d["waves_m"], d["twg_n"] // d["waves_n"], d["twg_k"]],
        pipeline_strategy=d["ps"],
        num_wg_per_cu=nwgcu(d, _HW),
    )


def make_sweep_grid(check_regs: bool = True) -> SweepGrid:
    grid = SweepGrid()
    add_gemm_sweep_axes(grid, _HW)

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
    args = parser.parse_args()

    pins = make_sweep_pins(args, GEMM_SWEEP_PIN_MAP)
    pins = resolve_derived_pins(pins or {})

    grid = make_sweep_grid(check_regs=not getattr(args, "no_reg_filter", False))
    if "_wg_m" in (pins or {}):
        target = pins.pop("_wg_m")
        grid.filter("waves_m", "waves_n", "occ", check=lambda d, t=target: wg_m(d, _HW) == t)

    all_configs, total = grid.generate(
        pins=pins or None,
        sample_size=getattr(args, "compile_sample", 4096)
    )

    results = bench_perf_sweep_pipelined(
        configs=all_configs,
        compile_fn=compile_multitile_gemm,
        repro_cmd_fn=_repro_cmd,
        num_gpus=args.num_gpus,
        compile_workers=args.compile_workers,
        compile_timeout=getattr(args, "compile_timeout", 60),
        post_compile_filter=fits_on_cu_post_compile,
        exec_sample=getattr(args, "exec_sample", 2000),
        zero_init=args.zero_init,
    )
    results, hsaco_map = results
    verify_top_configs(results, hsaco_map, _repro_cmd, top_n=50, num_gpus=args.num_gpus, label="102")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
