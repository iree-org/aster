"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (32x32x8 MFMA).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Usage (partial sweep / full sweep):
    python contrib/kittens/test/bench/bench_perf_sweep_002_gemm_fp16_weak_scaled.py --sweep
    python contrib/kittens/test/bench/bench_perf_sweep_002_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python contrib/kittens/test/bench/bench_perf_sweep_002_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config compile + run):
    python contrib/kittens/test/bench/bench_perf_sweep_002_gemm_fp16_weak_scaled.py \
        --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
        --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --stages 2 --k-scaling-factor 256

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep by default.
TOP_K_TO_RUN = [
    "m4864xn4096xk4096_wg38x32_w2x2_twg4x4x1_s2",
    "m4864xn4096xk8192_wg38x32_w2x2_twg4x4x1_s2",
    "m2432xn6144xk8192_wg38x32_w2x2_twg2x6x1_s2",
    "m4864xn4096xk2048_wg38x32_w2x2_twg4x4x1_s2",
    "m5472xn2048xk8192_wg19x16_w3x2_twg9x4x1_s2",
    "m7296xn2048xk8192_wg38x32_w2x2_twg6x2x1_s2",
    "m3648xn4096xk8192_wg19x16_w3x4_twg6x8x1_s2",
    "m7296xn6144xk8192_wg38x32_w3x2_twg6x6x1_s2",
    "m7296xn8192xk8192_wg38x32_w3x4_twg6x8x1_s2",
    "m3648xn4096xk4096_wg19x16_w3x4_twg6x8x1_s2", 
]

# Known-broken configs: add labels here to skip them during the sweep.
KNOWN_BROKEN = []

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_002_gemm_fp16_weak_scaled import (
    KERNEL_NAME,
    WeakScaleConfig,
    compile_weak_scaled_gemm,
    execute_weak_scaled_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    bench_perf_sweep,
    run_single,
    NUM_ITERATIONS,
)

# Sweep grid -- more conservative than perf_001 due to 16 VGPRs per C tile.
STAGE_CONFIGS = [2, 3, 4, 5]
WAVE_CONFIGS = [(2, 2), (3, 2), (3, 4), (4, 4)]
# Per-workgroup tile counts. Per-wave tiles derived as m_tiles_wg // m_waves.
# Max 1-3x multiples (was 1-5x for 16x16) since each C tile is 16 VGPRs.
_MULTIPLES = range(1, 4)
_K_TILES_RANGE = range(1, 4)
_tile_wg_pairs = {
    (mw * mm, nw * nm)
    for (mw, nw), mm, nm in itertools.product(WAVE_CONFIGS, _MULTIPLES, _MULTIPLES)
}
TILE_WG_CONFIGS = sorted((m, n, k) for m, n in _tile_wg_pairs for k in _K_TILES_RANGE)
WG_GRIDS = [(19, 16), (38, 32)]
# K = k_scaling_factor * k_tiles * 32 (each K tile = 32 elements for 32x32 transfer tiles).
# Lower factors than perf_001 since each tile covers 4x more K.
K_SCALING_FACTORS = [64, 128, 256]
SKIP_FIRST_N_CONFIGS = 0


def _generate_configs():
    """Generate the full sweep grid, filtering for divisibility."""
    return [
        WeakScaleConfig(
            m_wg, n_wg, m_w, n_w, m_twg, n_twg, k_t, stages, k_factor * k_t * 32
        )
        for k_factor in K_SCALING_FACTORS
        for m_wg, n_wg in WG_GRIDS
        for m_w, n_w in WAVE_CONFIGS
        for m_twg, n_twg, k_t in TILE_WG_CONFIGS
        for stages in STAGE_CONFIGS
        if m_twg % m_w == 0 and n_twg % n_w == 0
    ]


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    return (
        f"python bench/bench_perf_sweep_002_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles-wg {cfg.m_tiles_wg} --n-tiles-wg {cfg.n_tiles_wg} --k-tiles {cfg.k_tiles}"
        f" --stages {cfg.num_stages} --k-scaling-factor {k_factor}"
        f" --iterations {num_iterations}"
    )


def _cfg_to_cli_args(cfg):
    """Serialize config to CLI args for subprocess invocation."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    return [
        "--m-wg",
        str(cfg.m_wg),
        "--n-wg",
        str(cfg.n_wg),
        "--m-waves",
        str(cfg.m_waves),
        "--n-waves",
        str(cfg.n_waves),
        "--m-tiles-wg",
        str(cfg.m_tiles_wg),
        "--n-tiles-wg",
        str(cfg.n_tiles_wg),
        "--k-tiles",
        str(cfg.k_tiles),
        "--stages",
        str(cfg.num_stages),
        "--k-scaling-factor",
        str(k_factor),
    ]


def _make_config_from_args(args):
    """Construct a WeakScaleConfig from parsed CLI args."""
    k = args.k_scaling_factor * args.k_tiles * 32
    return WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles_wg,
        args.n_tiles_wg,
        args.k_tiles,
        args.stages,
        k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weak-scaled 32x32 GEMM benchmark: sweep or single-config repro",
    )
    add_sweep_cli_args(parser)
    # Single-config args
    parser.add_argument("--m-wg", type=int, help="Workgroups along M")
    parser.add_argument("--n-wg", type=int, help="Workgroups along N")
    parser.add_argument("--m-waves", type=int, help="Waves per WG along M")
    parser.add_argument("--n-waves", type=int, help="Waves per WG along N")
    parser.add_argument("--m-tiles-wg", type=int, help="Tiles per workgroup along M")
    parser.add_argument("--n-tiles-wg", type=int, help="Tiles per workgroup along N")
    parser.add_argument("--k-tiles", type=int, help="Tiles per wave along K")
    parser.add_argument("--stages", type=int, help="Pipeline stages")
    parser.add_argument(
        "--k-scaling-factor",
        type=int,
        help="K scaling factor (K = factor * k_tiles * 32)",
    )
    add_single_cli_args(parser)

    args = parser.parse_args()
    if args.full_sweep or args.sweep:
        bench_perf_sweep(
            configs=_generate_configs(),
            compile_fn=compile_weak_scaled_gemm,
            cfg_to_cli_args=_cfg_to_cli_args,
            repro_cmd_fn=_repro_cmd,
            script_path=__file__,
            top_k_to_run=TOP_K_TO_RUN,
            known_broken=KNOWN_BROKEN,
            skip_first_n=SKIP_FIRST_N_CONFIGS,
            full_sweep=args.full_sweep,
            num_gpus=args.num_gpus,
            compile_workers=args.compile_workers,
            kernel_name=KERNEL_NAME,
        )
    else:
        required = [
            "m_wg",
            "n_wg",
            "m_waves",
            "n_waves",
            "m_tiles_wg",
            "n_tiles_wg",
            "k_tiles",
            "stages",
            "k_scaling_factor",
        ]
        missing = [a for a in required if getattr(args, a) is None]
        if missing:
            flags = ", ".join(f"--{a.replace('_', '-')}" for a in missing)
            parser.error(f"Single-config mode requires: {flags}")
        run_single(
            _make_config_from_args(args),
            compile_weak_scaled_gemm,
            args,
            kernel_name=KERNEL_NAME,
            execute_fn=execute_weak_scaled_hsaco,
        )
