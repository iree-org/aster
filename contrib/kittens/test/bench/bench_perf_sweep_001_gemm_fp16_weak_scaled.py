"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (16x16x16 MFMA + dwordx4).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Usage (partial sweep / full sweep):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config compile + run):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
        --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --stages 2 --k-scaling-factor 128

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep (need to populate after first sweep).
TOP_K_TO_RUN = [
    "m4864xn4096xk4096_wg38x32_w2x2_twg8x8x1_s2",
    "m3648xn5120xk8192_wg38x32_w2x2_twg6x10x1_s2",
    "m4864xn4096xk8192_wg38x32_w2x2_twg8x8x1_s2",
    "m6080xn3072xk4096_wg38x32_w2x2_twg10x6x1_s2",
    "m6080xn3072xk8192_wg38x32_w2x2_twg10x6x1_s2",
    "m3648xn5120xk4096_wg38x32_w2x2_twg6x10x1_s2",
    "m4560xn2560xk8192_wg19x16_w3x2_twg15x10x1_s2",
    "m3648xn3072xk8192_wg38x32_w2x2_twg6x6x1_s2",
    "m4864xn4096xk2048_wg38x32_w2x2_twg8x8x1_s2",
    "m4864xn3072xk8192_wg38x32_w2x2_twg8x6x1_s2",
    "m3648xn4096xk8192_wg38x32_w2x2_twg6x8x1_s2",
    "m2432xn6144xk4096_wg38x32_w2x2_twg4x12x1_s2",
    "m3648xn4096xk4096_wg38x32_w2x2_twg6x8x1_s2",
    "m9120xn5120xk8192_wg38x32_w3x2_twg15x10x1_s2",
    "m6080xn3072xk2048_wg38x32_w2x2_twg10x6x1_s2",
    "m7296xn4096xk4096_wg38x32_w2x2_twg12x8x1_s2",
    "m4864xn6144xk4096_wg38x32_w2x2_twg8x12x1_s2",
    "m7296xn4096xk2048_wg38x32_w2x2_twg12x8x1_s2",
    "m6080xn2048xk4096_wg38x32_w2x2_twg10x4x1_s2",
    "m7296xn6144xk8192_wg38x32_w3x2_twg12x12x1_s2",
]

# Known-broken configs: add labels here to skip them during the sweep.
KNOWN_BROKEN = []

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
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

# Sweep grid -- 16x16 MFMA with dwordx4: 4 VGPRs per C tile (vs 16 for 32x32).
# More tiles feasible per wave, so wider multiples than 32x32 variant.
STAGE_CONFIGS = [2, 3, 4, 5]
WAVE_CONFIGS = [(2, 2), (3, 2), (3, 4), (4, 4)]
# Per-workgroup tile counts. Per-wave tiles derived as m_tiles_wg // m_waves.
# Max 1-5x multiples: 4 VGPRs per C tile allows more tiles per wave.
_MULTIPLES = range(1, 6)
_K_TILES_RANGE = range(1, 4)
_tile_wg_pairs = {
    (mw * mm, nw * nm)
    for (mw, nw), mm, nm in itertools.product(WAVE_CONFIGS, _MULTIPLES, _MULTIPLES)
}
TILE_WG_CONFIGS = sorted((m, n, k) for m, n in _tile_wg_pairs for k in _K_TILES_RANGE)
WG_GRIDS = [(19, 16), (38, 32)]
# K = k_scaling_factor * k_tiles * 32 (each 16x32 transfer tile = 32 K elements).
K_SCALING_FACTORS = [64, 128, 256]
SKIP_FIRST_N_CONFIGS = 0


MIN_DIM = 2048  # Skip configs where M, N, or K < 2048


def _generate_configs():
    """Generate the full sweep grid, filtering for divisibility and minimum dimensions."""
    configs = []
    for k_factor in K_SCALING_FACTORS:
        for m_wg, n_wg in WG_GRIDS:
            for m_w, n_w in WAVE_CONFIGS:
                for m_twg, n_twg, k_t in TILE_WG_CONFIGS:
                    if m_twg % m_w != 0 or n_twg % n_w != 0:
                        continue
                    for stages in STAGE_CONFIGS:
                        k = k_factor * k_t * 32
                        cfg = WeakScaleConfig(
                            m_wg, n_wg, m_w, n_w, m_twg, n_twg, k_t, stages, k
                        )
                        if (
                            cfg.m_dim < MIN_DIM
                            or cfg.n_dim < MIN_DIM
                            or cfg.k < MIN_DIM
                        ):
                            continue
                        configs.append(cfg)
    return configs


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    return (
        f"python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
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


CORRECTNESS_K = 128  # Small K for fast compile+execute correctness checks.
CORRECTNESS_TOP_N = 20  # Number of top configs to verify after a sweep.


def verify_top_configs(results, num_configs=CORRECTNESS_TOP_N):
    """Phase 3: Verify correctness of the top N configs from the sweep.

    Recompiles each config at K=128 (fast), executes, and checks against numpy.
    """
    import tempfile
    import numpy as np

    if not results:
        return
    top = results[:num_configs]
    print(
        f"\n--- Phase 3: Correctness verification (top {len(top)} configs, K={CORRECTNESS_K}) ---"
    )
    sys.stdout.flush()

    passed = 0
    failed_labels = []
    for rank, (cfg, ms, tflops, pct) in enumerate(top, 1):
        small_cfg = WeakScaleConfig(
            cfg.m_wg,
            cfg.n_wg,
            cfg.m_waves,
            cfg.n_waves,
            cfg.m_tiles_wg,
            cfg.n_tiles_wg,
            cfg.k_tiles,
            cfg.num_stages,
            CORRECTNESS_K,
        )
        tag = f"[{rank}/{len(top)}] {cfg.label}"
        try:
            np.random.seed(42)
            A = (np.random.randn(small_cfg.m_dim, small_cfg.k) * 0.1).astype(np.float16)
            B = (np.random.randn(small_cfg.n_dim, small_cfg.k) * 0.1).astype(np.float16)
            with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
                compile_weak_scaled_gemm(small_cfg, tmp.name)
                C_output, _ = execute_weak_scaled_hsaco(
                    small_cfg, tmp.name, 1, A, B, skip_gpu_check=True
                )
            expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
            np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
            passed += 1
            print(f"  PASS  {tag}")
        except Exception as e:
            failed_labels.append(cfg.label)
            err_line = str(e).split("\n")[0][:120]
            print(f"  FAIL  {tag}: {err_line}")
        sys.stdout.flush()

    print(f"\nCorrectness: {passed}/{len(top)} passed", end="")
    if failed_labels:
        print(f", {len(failed_labels)} FAILED:")
        for label in failed_labels:
            print(f"  {label}")
    else:
        print(" -- all correct")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weak-scaled 16x16+dwordx4 GEMM benchmark: sweep or single-config repro",
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
        help="K scaling factor (K = factor * k_tiles * 32, each 16x32 tile = 32 K elements)",
    )
    add_single_cli_args(parser)

    args = parser.parse_args()
    if args.full_sweep or args.sweep:
        results = bench_perf_sweep(
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
        verify_top_configs(results)
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
