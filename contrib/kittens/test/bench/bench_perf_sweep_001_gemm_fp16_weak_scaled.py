"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (16x16x16 MFMA + dwordx4).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Sweep axes: load_type (flat/buffer) x a_path (lds/direct) x unroll_multiplier (1,2,3).
By default sweeps all implemented (a_path, load_type) combos.

Usage (sweep):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-buffer   # buffer only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-flat     # flat only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --direct-a     # direct-A only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
        --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --stages 2 --k-scaling-factor 128
    ... --use-flat      # flat memory ops (default)
    ... --use-buffer    # buffer memory ops
    ... --direct-a      # A via bpermute (LDS bypass)

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep (need to populate after first sweep).
# Label suffix scheme: _flat, _buf (LDS path), _direct_flat, _direct_buf (direct-A path).
_TOP_K_BASE = [
    "m3648xn4096xk4096_wg38x16_w2x2_twg6x16x1_s2_direct_flat",
    "m4864xn4096xk8192_wg38x32_w2x2_twg8x8x1_s2_direct_flat",
    "m3648xn8192xk8192_wg19x32_w2x2_twg12x16x1_s2_direct_flat",
    "m3040xn16384xk4096_wg19x64_w2x4_twg10x16x1_s2_buf",
    "m4864xn2048xk8192_wg38x32_w4x1_twg8x4x1_s4_direct_flat",
    "m7296xn2048xk4096_wg19x16_w4x2_twg24x8x1_s2_flat",
    "m4560xn8192xk4096_wg19x64_w3x4_twg15x8x1_s2_flat",
    "m3040xn16384xk4096_wg19x64_w2x4_twg10x16x1_s3_direct_flat",
    "m3648xn4096xk4096_wg19x32_w6x2_twg12x8x1_s3_buf",
    "m6080xn2048xk8192_wg19x16_w2x2_twg20x8x1_s2_flat",
    "m9728xn4096xk2048_wg76x64_w2x2_twg8x4x1_s3_direct_flat",
    "m3040xn16384xk8192_wg19x64_w1x4_twg10x16x1_s3_direct_flat",
    "m2432xn8192xk8192_wg19x64_w2x4_twg8x8x1_s2_flat",
    "m4864xn4096xk8192_wg38x16_w1x4_twg8x16x1_s2_direct_flat",
    "m2432xn2048xk8192_wg19x16_w2x4_twg8x8x2_s2_flat",
    "m2432xn4096xk8192_wg19x32_w2x4_twg8x8x1_s4_buf",
    "m2432xn4096xk4096_wg38x32_w1x4_twg4x8x1_s4_direct_flat",
    "m2432xn8192xk16384_wg38x64_w2x2_twg4x8x2_s2_direct_flat",
    "m9728xn2048xk4096_wg38x16_w2x2_twg16x8x1_s2_direct_flat",
    "m3040xn2048xk2048_wg19x16_w2x2_twg10x8x1_s3_flat",
]


import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
    MLIR_FILES,
    WeakScaleConfig,
    compile_gemm,
    execute_gemm_hsaco,
)
from bench_harness import (
    add_sweep_cli_args,
    add_single_cli_args,
    bench_perf_sweep,
    make_sweep_filter,
    run_single,
)

# Sweep grid -- 16x16 MFMA with dwordx4: 4 VGPRs per C tile (vs 16 for 32x32).
# More tiles feasible per wave, so wider multiples than 32x32 variant.
STAGE_CONFIGS = [2, 3, 4, 5]
# Wave configs: multiples-of-4 wave counts split across MxN.
# n_waves must be a power of 2 (delinearize from 1-D block ID).
_WAVE_BASES = [(1, 4), (2, 2), (4, 1)]
_is_po2 = lambda x: x > 0 and (x & (x - 1)) == 0
WAVE_CONFIGS = sorted(
    {
        (bm * k1, bn * k2)
        for bm, bn in _WAVE_BASES
        for k1 in range(1, 7)
        for k2 in range(1, 7)
        if bm * k1 <= 6
        and _is_po2(bn * k2)
        and bn * k2 <= 8
        and bm * k1 * bn * k2 <= 16
        and (bm * k1 * bn * k2) % 4 == 0
    }
)
# Per-workgroup tile counts. Per-wave tiles derived as m_tiles_wg // m_waves.
# N-dimension multiples must be powers of 2 (delinearize from 1-D block ID).
_M_MULTIPLES = range(1, 6)
_N_MULTIPLES = [1, 2, 4]  # powers of 2
_K_TILES_RANGE = range(1, 4)
_tile_wg_pairs = {
    (mw * mm, nw * nm)
    for (mw, nw), mm, nm in itertools.product(WAVE_CONFIGS, _M_MULTIPLES, _N_MULTIPLES)
}
TILE_WG_CONFIGS = sorted((m, n, k) for m, n in _tile_wg_pairs for k in _K_TILES_RANGE)
_WG_BASE = (19, 16)
_NUM_SIMDS = 4
# Occupancy targets = desired waves per SIMD. From this + the wave config we
# derive num_wg_per_cu and the M-dimension WG multiplier. See _generate_configs.
OCCUPANCY_TARGETS = [1, 2, 3, 4]
# N-dimension workgroup multipliers (independent of occupancy, for problem size variety).
N_WG_MULTIPLIERS = [1, 2, 4]  # must be powers of 2
# K = k_scaling_factor * k_tiles * 32 (each 16x32 transfer tile = 32 K elements).
K_SCALING_FACTORS = [64, 128, 256]
# LCM unroll on/off sweep. When True, also sweeps unroll multipliers.
LCM_UNROLL_CONFIGS = [True, False]
# Unroll factor multipliers: scale the LCM unroll factor by this amount.
# Only swept when lcm_unroll=True; pinned to [1] when False.
UNROLL_MULTIPLIERS = [1, 2, 3]
# Epilogue peeling: fully unroll cleanup loop after LCM unrolling.
EPILOGUE_PEELING_CONFIGS = [True, False]

MIN_DIM = 2000  # Skip configs where M, N, or K < 3000


def _precompile_reject_reason(cfg, check_regs=True):
    """Return rejection reason string, or None if config passes pre-compile filter."""
    from aster.compiler.metadata import KernelResources

    est = KernelResources(
        vgpr_count=cfg.estimated_vgprs if check_regs else 0,
        agpr_count=cfg.estimated_agprs if check_regs else 0,
        lds_bytes=cfg.lds_bytes,
    )
    violations = est.check_occupancy(cfg.num_threads)
    return violations[0] if violations else None


def fits_on_cu_post_compile(cfg, res):
    """Post-compilation check: can this config launch given actual resource usage?

    Delegates entirely to check_occupancy (registers + LDS).
    Returns True if launchable, False otherwise (prints violations).
    """
    violations = res.check_occupancy(cfg.num_threads)
    if violations:
        for v in violations:
            print(f"  OCCUPANCY ERROR [{cfg.label}]: {v}")
        return False
    return True


def _make_label_suffix(a_path, load_type):
    """Build label suffix from a_path and load_type, e.g. '_flat', '_buf', '_direct_flat'."""
    lt = "buf" if load_type == "buffer" else "flat"
    return f"_direct_{lt}" if a_path == "direct" else f"_{lt}"


def _generate_configs(
    variants=None, sample_size=3000, check_regs=True, sweep_filter=None
):
    """Generate the full sweep grid, filtering for divisibility and minimum dimensions.

    Args:
        variants: list of (a_path, load_type) tuples to sweep.
            Defaults to all implemented combos from MLIR_FILES.
        sample_size: If > 0, randomly sample this many configs from the full grid.
            Set to 0 to return all configs.
        check_regs: If True, pre-filter configs whose estimated VGPR/AGPR usage
            exceeds the occupancy-derived register budget.
        sweep_filter: Optional predicate (cfg) -> bool to filter configs before
            sampling. Used by CLI dimension-pinning (e.g. --n-waves 4).
    """
    import math
    import random

    if variants is None:
        variants = list(MLIR_FILES.keys())
    configs = []
    filtered = []
    for a_path, load_type in variants:
        if (a_path, load_type) not in MLIR_FILES:
            continue
        suffix = _make_label_suffix(a_path, load_type)
        for k_factor in K_SCALING_FACTORS:
            for m_w, n_w in WAVE_CONFIGS:
                num_waves = m_w * n_w
                waves_per_simd = math.ceil(num_waves / _NUM_SIMDS)
                for occ_target in OCCUPANCY_TARGETS:
                    # Derive num_wg_per_cu from occupancy target.
                    if occ_target % waves_per_simd != 0:
                        continue
                    num_wg_per_cu = occ_target // waves_per_simd
                    # M workgroups scale with num_wg_per_cu.
                    m_wg = _WG_BASE[0] * num_wg_per_cu
                    for n_mult in N_WG_MULTIPLIERS:
                        n_wg = _WG_BASE[1] * n_mult
                        for m_twg, n_twg, k_t in TILE_WG_CONFIGS:
                            if m_twg % m_w != 0 or n_twg % n_w != 0:
                                continue
                            for stages in STAGE_CONFIGS:
                                for lcm in LCM_UNROLL_CONFIGS:
                                    for um in (UNROLL_MULTIPLIERS if lcm else [1]):
                                        for ep in EPILOGUE_PEELING_CONFIGS:
                                            k = k_factor * k_t * 32
                                            cfg = WeakScaleConfig(
                                                m_wg,
                                                n_wg,
                                                m_w,
                                                n_w,
                                                m_twg,
                                                n_twg,
                                                k_t,
                                                stages,
                                                k,
                                                load_type=load_type,
                                                a_path=a_path,
                                                num_wg_per_cu=num_wg_per_cu,
                                                lcm_unroll=lcm,
                                                unroll_factor_multiplier=um,
                                                epilogue_peeling=ep,
                                                _label_suffix=suffix,
                                            )
                                            if (
                                                cfg.m_dim < MIN_DIM
                                                or cfg.n_dim < MIN_DIM
                                                or cfg.k < MIN_DIM
                                            ):
                                                continue
                                            reason = _precompile_reject_reason(
                                                cfg, check_regs=check_regs
                                            )
                                            if reason is not None:
                                                filtered.append((cfg.label, reason))
                                                continue
                                            configs.append(cfg)

    # Save filtered configs to temp file.
    if filtered:
        import tempfile as _tmp

        fd, filt_path = _tmp.mkstemp(
            prefix="bench_filtered_", suffix=".txt", dir="/tmp"
        )
        with os.fdopen(fd, "w") as f:
            for label, reason in filtered:
                f.write(f"{label}: {reason}\n")
        print(
            f"{len(filtered)} configs skipped by pre-compile filter "
            f"(details in {filt_path})"
        )

    if sweep_filter:
        before = len(configs)
        configs = [c for c in configs if sweep_filter(c)]
        print(f"Sweep filter: {before} -> {len(configs)} eligible configs")

    total = len(configs)
    n = min(sample_size, total) if sample_size > 0 else total
    if n < total:
        configs = random.sample(configs, n)
    print(f"Compiling {n} / {total} eligible configs")
    return configs


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    buf_flag = " --use-buffer" if cfg.use_buffer else " --use-flat"
    direct_flag = " --direct-a" if cfg.direct_a else ""
    lcm_flag = "" if cfg.lcm_unroll else " --no-lcm-unroll"
    um_flag = (
        f" --unroll-multiplier {cfg.unroll_factor_multiplier}"
        if cfg.unroll_factor_multiplier > 1
        else ""
    )
    peel_flag = "" if cfg.epilogue_peeling else " --no-epilogue-peeling"
    wg_per_cu_flag = (
        f" --num-wg-per-cu {cfg.num_wg_per_cu}" if cfg.num_wg_per_cu != 1 else ""
    )
    return (
        f"python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles-wg {cfg.m_tiles_wg} --n-tiles-wg {cfg.n_tiles_wg} --k-tiles {cfg.k_tiles}"
        f" --stages {cfg.num_stages} --k-scaling-factor {k_factor}"
        f"{buf_flag}{direct_flag}{lcm_flag}{um_flag}{peel_flag}{wg_per_cu_flag}"
        f" --iterations {num_iterations}"
    )


def _make_config_from_args(args, load_type, a_path):
    """Construct a WeakScaleConfig from parsed CLI args."""
    k = args.k_scaling_factor * args.k_tiles * 32
    suffix = _make_label_suffix(a_path, load_type)
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
        load_type=load_type,
        a_path=a_path,
        num_wg_per_cu=getattr(args, "num_wg_per_cu", 1) or 1,
        lcm_unroll=getattr(args, "lcm_unroll", True),
        unroll_factor_multiplier=getattr(args, "unroll_multiplier", 1) or 1,
        epilogue_peeling=getattr(args, "epilogue_peeling", True),
        _label_suffix=suffix,
    )


def _compile_fn(cfg, output_hsaco_path, **kwargs):
    """Compile wrapper -- cfg carries load_type, a_path, unroll and peeling config."""
    return compile_gemm(
        cfg,
        output_hsaco_path,
        unroll_factor_multiplier=cfg.unroll_factor_multiplier,
        epilogue_peeling=cfg.epilogue_peeling,
        **kwargs,
    )


CORRECTNESS_K = 2048  # Small K for fast compile+execute correctness checks.
CORRECTNESS_TOP_N = 100  # Number of top configs to verify after a sweep.


def verify_top_configs(
    results, hsaco_paths, num_configs=CORRECTNESS_TOP_N, num_gpus=None
):
    """Phase 3: Verify top N configs using same subprocess pattern as execution."""
    from bench_harness import (
        check_numpy_blas,
        _save_tmpfile,
        detect_num_gpus,
        verify_on_gpus,
    )

    if not results:
        return
    if num_gpus is None:
        num_gpus = detect_num_gpus()
    if num_gpus == 0:
        print("\nNo GPUs detected -- skipping correctness verification.")
        return
    top = results[:num_configs]
    to_verify = [c for c, *_ in top if c.label in hsaco_paths]
    if not to_verify:
        return
    print(
        f"\n--- Phase 3: Correctness ({len(to_verify)} configs, {num_gpus} GPU(s)) ---"
    )
    check_numpy_blas(label="correctness")

    passed, errors = verify_on_gpus(to_verify, hsaco_paths, num_gpus)

    print(f"\nCorrectness: {passed}/{len(to_verify)} passed", end="")
    if errors:
        path = _save_tmpfile("bench_verify_", errors)
        print(f", {len(errors)} FAILED (details in {path})")
    else:
        print(" -- all correct")


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
    buf_group = parser.add_mutually_exclusive_group()
    buf_group.add_argument(
        "--use-buffer",
        action="store_true",
        help="Sweep buffer_load/buffer_store only",
    )
    buf_group.add_argument(
        "--use-flat",
        action="store_true",
        help="Sweep global_load/global_store only",
    )
    parser.add_argument(
        "--direct-a",
        action="store_true",
        help="A operand via bpermute (LDS bypass) instead of LDS",
    )
    parser.add_argument(
        "--no-lcm-unroll",
        action="store_true",
        help="Disable LCM-based kernel loop unrolling",
    )
    parser.add_argument(
        "--unroll-multiplier",
        type=int,
        default=None,
        help="Unroll factor multiplier (scales LCM unroll factor, default: 1)",
    )
    parser.add_argument(
        "--no-epilogue-peeling",
        action="store_true",
        help="Disable epilogue peeling (keep cleanup loop after LCM unrolling)",
    )
    parser.add_argument(
        "--desired-simd-occupancy",
        type=int,
        default=None,
        help="Filter sweep to configs with this SIMD occupancy (waves per SIMD)",
    )

    args = parser.parse_args()
    args.lcm_unroll = not args.no_lcm_unroll
    args.epilogue_peeling = not args.no_epilogue_peeling

    # Determine a_path
    a_path = "direct" if args.direct_a else "lds"

    # Determine load_type variants to sweep.
    if args.use_buffer:
        load_types = ["buffer"]
    elif args.use_flat:
        load_types = ["flat"]
    else:
        load_types = ["flat", "buffer"]

    # Build (a_path, load_type) variant list.
    # In sweep mode without --direct-a, sweep all implemented combos.
    # With --direct-a, sweep only direct combos.
    if args.full_sweep or args.sweep:
        if args.direct_a:
            variants = [(a_path, lt) for lt in load_types]
        else:
            # Sweep all a_path values for each load_type
            variants = [(ap, lt) for lt in load_types for ap in ["lds", "direct"]]
    else:
        variants = [(a_path, lt) for lt in load_types]

    # Filter to implemented combos
    variants = [(ap, lt) for ap, lt in variants if (ap, lt) in MLIR_FILES]

    # For single-config mode
    load_type = "buffer" if args.use_buffer else "flat"

    # TOP_K labels include suffix -- filter to selected variants.
    variant_suffixes = {_make_label_suffix(ap, lt) for ap, lt in variants}
    top_k_to_run = [
        label
        for label in _TOP_K_BASE
        if any(label.endswith(s) for s in variant_suffixes)
    ]

    if args.full_sweep or args.sweep:
        variant_str = ", ".join(f"{ap}/{lt}" for ap, lt in variants)
        print(f"Variants: {variant_str}")

        # Pin sweep dimensions from CLI args (e.g. --n-waves 4 filters the grid).
        _SWEEP_ATTR_MAP = {
            "m_wg": "m_wg",
            "n_wg": "n_wg",
            "m_waves": "m_waves",
            "n_waves": "n_waves",
            "m_tiles_wg": "m_tiles_wg",
            "n_tiles_wg": "n_tiles_wg",
            "k_tiles": "k_tiles",
            "stages": "num_stages",
            "k_scaling_factor": "k_scaling_factor",
            "unroll_multiplier": "unroll_factor_multiplier",
            "desired_simd_occupancy": "simd_occupancy",
        }
        sweep_filter = make_sweep_filter(args, _SWEEP_ATTR_MAP)

        all_configs = _generate_configs(
            variants,
            sample_size=getattr(args, "compile_sample", 4096),
            check_regs=not getattr(args, "no_reg_filter", False),
            sweep_filter=sweep_filter,
        )

        def _post_compile_filter(cfg, res):
            """Post-compilation filter: reject configs exceeding VGPR or LDS limits."""
            return fits_on_cu_post_compile(cfg, res)

        results = bench_perf_sweep(
            configs=all_configs,
            compile_fn=_compile_fn,
            repro_cmd_fn=_repro_cmd,
            top_k_to_run=top_k_to_run,
            full_sweep=args.full_sweep,
            num_gpus=args.num_gpus,
            compile_workers=args.compile_workers,
            compile_timeout=getattr(args, "compile_timeout", 60),
            post_compile_filter=_post_compile_filter,
            exec_sample=getattr(args, "exec_sample", 2000),
        )
        results, hsaco_map = results
        verify_top_configs(results, hsaco_map, num_gpus=args.num_gpus)
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
            _make_config_from_args(args, load_type, a_path),
            compile_gemm,
            args,
            execute_fn=execute_gemm_hsaco,
        )
