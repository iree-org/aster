"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM (16x16x16 MFMA + dwordx4).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Sweep axes: load_type (flat/buffer) x b_path (lds/direct) x unroll_multiplier (1,2,3).
By default sweeps all implemented (b_path, load_type) combos.

Usage (sweep):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-buffer   # buffer only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --use-flat     # flat only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --direct-b     # direct-B only
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config):
    python .../bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 38 --n-wg 32 --m-waves 2 --n-waves 2 \
        --m-tiles-wg 4 --n-tiles-wg 4 --k-tiles 1 --a-stages 2 --k-scaling-factor 128
    ... --use-flat      # flat memory ops (default)
    ... --use-buffer    # buffer memory ops
    ... --direct-b      # B via bpermute (LDS bypass)

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep (need to populate after first sweep).
# Label suffix scheme: _flat, _buf (LDS path), _direct_flat, _direct_buf (direct-B path).
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
    make_sweep_pins,
    run_single,
)

# --- GPU hardware constants ---
# Query from HIP at runtime when available, fall back to gfx942 defaults for
# cross-compilation (macOS). Source: aster.core.device -> hipDeviceProp_t.
try:
    from aster.core.device import try_query_device

    _dev = try_query_device(0)
except ImportError:
    _dev = None

# Per-SIMD register file size (arch VGPRs, 512 on gfx942).
GPU_VGPRS_PER_SIMD = _dev.vgprs_per_simd if _dev else 512
# Max addressable VGPRs per wave (256 on gfx942).
GPU_MAX_VGPRS = min(GPU_VGPRS_PER_SIMD, 256)
# Max AGPRs per wave (same as VGPRs on CDNA).
GPU_MAX_AGPRS = GPU_MAX_VGPRS
# LDS per CU (bytes, 65536 on gfx942, 262144 on gfx950).
GPU_LDS_PER_CU = _dev.lds_per_cu if _dev else 65536
# VGPR allocation granularity.
GPU_VGPR_GRANULE = _dev.vgpr_alloc_granule if _dev else 8


# Sweep grid -- 16x16 MFMA with dwordx4: 4 VGPRs per C tile (vs 16 for 32x32).
# More tiles feasible per wave, so wider multiples than 32x32 variant.
# Pipeline strategies to sweep. Each integer selects from PIPELINE_STRATEGIES
# in kittens_helpers.py (0=no pipeline, 10=max depth). Higher strategies use
# more VGPRs and LDS, so fewer configs pass the resource filter.
PIPELINE_STRATEGY_CONFIGS = [1, 3, 5, 6, 7, 9]
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
_K_TILES_RANGE = range(1, 9)
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
# Low-level instruction scheduler on/off.
LL_SCHED_CONFIGS = [True, False]
# Hoist iter_arg waits to loop head on/off.
HOIST_WAIT_CONFIGS = [True, False]

MIN_DIM = 2000  # Skip configs where M, N, or K < 3000


def fits_on_cu_post_compile(cfg, res):
    """Post-compilation check: can this config launch given actual resource usage?

    Delegates entirely to check_occupancy (registers + LDS).
    Returns True if launchable, False otherwise (prints violations).
    """
    violations = res.check_occupancy(cfg.num_threads, num_wg_per_cu=cfg.num_wg_per_cu)
    if violations:
        for v in violations:
            print(f"  OCCUPANCY ERROR [{cfg.label}]: {v}")
        return False
    return True


def _make_label_suffix(b_path, load_type):
    """Build label suffix from b_path and load_type, e.g. '_flat', '_buf', '_direct_flat'."""
    lt = "buf" if load_type == "buffer" else "flat"
    return f"_direct_{lt}" if b_path == "direct" else f"_{lt}"


def _passes_resource_check(mw, nw, mtwg, ntwg, kt, a_stg, b_stg, nwgcu, is_direct):
    """Check LDS and VGPR limits without constructing a WeakScaleConfig."""
    num_waves = mw * nw
    mt, nt = mtwg // mw, ntwg // nw
    ceildiv = lambda a, b: (a + b - 1) // b

    # LDS.
    a_lds = a_stg * mtwg * kt * 1024
    b_lds = 0 if is_direct else b_stg * ntwg * kt * 1024
    if a_lds + b_lds > GPU_LDS_PER_CU // max(nwgcu, 1):
        return False

    # VGPR estimate (same formulas as WeakScaleConfig.estimated_vgprs).
    waves_m = min(mtwg, num_waves)
    waves_k_a = max(1, num_waves // max(waves_m, 1))
    coop_a = ceildiv(mtwg, max(waves_m, 1)) * ceildiv(kt, max(waves_k_a, 1))
    a_load = coop_a * a_stg * 4
    a_read = mt * kt * 4
    if is_direct:
        b_load = nt * kt * b_stg * 4
        b_split, overhead = nt * kt * 4, 30
    else:
        waves_n = min(ntwg, num_waves)
        waves_k_b = max(1, num_waves // max(waves_n, 1))
        coop_b = ceildiv(ntwg, max(waves_n, 1)) * ceildiv(kt, max(waves_k_b, 1))
        b_load = coop_b * b_stg * 4
        b_split, overhead = nt * kt * 4, 10
    # 1.2x + 16 safety margin for spills/temporaries.
    est_v = (a_load + a_read + b_load + b_split + overhead) * 6 // 5 + 16
    est_a = mt * nt * 4

    if est_v > GPU_MAX_VGPRS or est_a > GPU_MAX_AGPRS:
        return False
    combined = est_v + est_a
    if combined > GPU_VGPRS_PER_SIMD:
        return False
    # Per-CU register file check (clr/rocclr/device/rocm/rocdevice.cpp:1604).
    aligned = ((combined + GPU_VGPR_GRANULE - 1) // GPU_VGPR_GRANULE) * GPU_VGPR_GRANULE
    total_waves = num_waves * nwgcu
    if aligned * total_waves * 64 > GPU_VGPRS_PER_SIMD * _NUM_SIMDS * 64:
        return False
    return True


def _generate_configs(
    variants=None, sample_size=3000, check_regs=True, sweep_pins=None
):
    """Generate eligible configs via nested loops with early rejection.

    Filters are applied hierarchically -- dimension checks first, then LDS/register
    checks -- so the inner loops only run for valid outer combos. Each variant gets an
    equal share of the sample budget.
    """
    import random
    from kittens_helpers import pipeline_strategy_stages

    if variants is None:
        variants = list(MLIR_FILES.keys())
    active = [(bp, lt) for bp, lt in variants if (bp, lt) in MLIR_FILES]
    if not active:
        return []

    per_variant = max(sample_size // len(active), 1) if sample_size > 0 else 0

    # Build the "flag" configs: all boolean/unroll combos as a flat list of tuples.
    flag_cfgs = [
        (lcm, um, ep, ll, hw)
        for lcm in LCM_UNROLL_CONFIGS
        for um in (UNROLL_MULTIPLIERS if lcm else [1])
        for ep in EPILOGUE_PEELING_CONFIGS
        for ll in LL_SCHED_CONFIGS
        for hw in HOIST_WAIT_CONFIGS
    ]

    strat_stages = {s: pipeline_strategy_stages(s) for s in PIPELINE_STRATEGY_CONFIGS}
    _pin = lambda key, val: (
        not sweep_pins or key not in sweep_pins or sweep_pins[key] == val
    )
    all_configs = []
    total_eligible = 0

    for vi, (b_path, load_type) in enumerate(active):
        suffix = _make_label_suffix(b_path, load_type)
        is_direct = b_path in ("direct_b", "direct_ab")
        eligible = []

        for mw, nw in WAVE_CONFIGS:
            if not (_pin("m_waves", mw) and _pin("n_waves", nw)):
                continue
            nwaves = mw * nw
            wps = (nwaves + _NUM_SIMDS - 1) // _NUM_SIMDS

            for occ in OCCUPANCY_TARGETS:
                if occ % wps != 0:
                    continue
                nwgcu = occ // wps
                m_wg = _WG_BASE[0] * nwgcu
                simd_occ = nwgcu * wps
                if not (_pin("m_wg", m_wg) and _pin("simd_occupancy", simd_occ)):
                    continue

                for n_mult in N_WG_MULTIPLIERS:
                    n_wg = _WG_BASE[1] * n_mult
                    if not _pin("n_wg", n_wg):
                        continue

                    for mtwg, ntwg, kt in TILE_WG_CONFIGS:
                        if (
                            mtwg % mw != 0
                            or ntwg % nw != 0
                            or mtwg < nwaves
                            or m_wg * mtwg * 16 < MIN_DIM
                            or n_wg * ntwg * 16 < MIN_DIM
                        ):
                            continue
                        if not (
                            _pin("m_tiles_wg", mtwg)
                            and _pin("n_tiles_wg", ntwg)
                            and _pin("k_tiles", kt)
                        ):
                            continue

                        for strategy in PIPELINE_STRATEGY_CONFIGS:
                            a_stg, b_stg = strat_stages[strategy]
                            depth = max(a_stg, b_stg)
                            if depth > a_stg:
                                continue
                            if not _pin("pipeline_strategy", strategy):
                                continue
                            if check_regs and not _passes_resource_check(
                                mw,
                                nw,
                                mtwg,
                                ntwg,
                                kt,
                                a_stg,
                                b_stg,
                                nwgcu,
                                is_direct,
                            ):
                                continue

                            for k_factor in K_SCALING_FACTORS:
                                k = k_factor * kt * 32
                                if k < MIN_DIM or k_factor <= depth:
                                    continue
                                if not _pin("k_scaling_factor", k_factor):
                                    continue

                                for lcm, um, ep, ll, hw in flag_cfgs:
                                    if not (
                                        _pin("unroll_factor_multiplier", um)
                                        and _pin("lcm_unroll", lcm)
                                        and _pin("epilogue_peeling", ep)
                                        and _pin("ll_sched", ll)
                                        and _pin("hoist_wait", hw)
                                    ):
                                        continue
                                    eligible.append(
                                        WeakScaleConfig(
                                            m_wg,
                                            n_wg,
                                            mw,
                                            nw,
                                            mtwg,
                                            ntwg,
                                            kt,
                                            a_stg,
                                            k,
                                            load_type=load_type,
                                            b_path=b_path,
                                            b_stages=b_stg,
                                            num_wg_per_cu=nwgcu,
                                            lcm_unroll=lcm,
                                            unroll_factor_multiplier=um,
                                            epilogue_peeling=ep,
                                            ll_sched=ll,
                                            hoist_wait=hw,
                                            pipeline_strategy=strategy,
                                            _label_suffix=suffix,
                                        )
                                    )

        n_eligible = len(eligible)
        total_eligible += n_eligible
        if per_variant > 0 and n_eligible > per_variant:
            eligible = random.sample(eligible, per_variant)
        all_configs.extend(eligible)
        print(
            f"  [{vi+1}/{len(active)}] {b_path}/{load_type}: "
            f"{n_eligible:,} eligible, {len(eligible):,} selected"
        )

    print(f"Total: {total_eligible:,} eligible, {len(all_configs):,} selected")
    return all_configs


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    buf_flag = " --use-buffer" if cfg.use_buffer else " --no-use-buffer"
    flat_flag = " --no-use-flat" if cfg.use_buffer else " --use-flat"
    direct_b_flag = " --direct-b" if cfg.direct_b else " --no-direct-b"
    direct_a_flag = " --direct-a" if cfg.direct_a else " --no-direct-a"
    lcm_flag = " --lcm-unroll" if cfg.lcm_unroll else " --no-lcm-unroll"
    um_flag = (
        f" --unroll-multiplier {cfg.unroll_factor_multiplier}"
        if cfg.unroll_factor_multiplier > 1
        else ""
    )
    peel_flag = (
        " --epilogue-peeling" if cfg.epilogue_peeling else " --no-epilogue-peeling"
    )
    wg_per_cu_flag = (
        f" --num-wg-per-cu {cfg.num_wg_per_cu}" if cfg.num_wg_per_cu != 1 else ""
    )
    ll_flag = " --ll-sched" if getattr(cfg, "ll_sched", False) else " --no-ll-sched"
    hw_flag = (
        " --hoist-wait" if getattr(cfg, "hoist_wait", False) else " --no-hoist-wait"
    )
    ps = getattr(cfg, "pipeline_strategy", -1)
    ps_flag = f" --pipeline-strategy {ps}" if ps >= 0 else ""
    return (
        f"python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles-wg {cfg.m_tiles_wg} --n-tiles-wg {cfg.n_tiles_wg} --k-tiles {cfg.k_tiles}"
        f" --pipeline-strategy {cfg.pipeline_strategy} --k-scaling-factor {k_factor}"
        f"{buf_flag}{flat_flag}{direct_b_flag}{direct_a_flag}"
        f"{lcm_flag}{um_flag}{peel_flag}{wg_per_cu_flag}"
        f"{ll_flag}{hw_flag}"
        f" --iterations {num_iterations}"
    )


def _make_config_from_args(args, load_type, b_path):
    """Construct a WeakScaleConfig from parsed CLI args."""
    k = args.k_scaling_factor * args.k_tiles * 32
    suffix = _make_label_suffix(b_path, load_type)
    # pipeline_strategy is the primary control; a_stages/b_stages derived in __post_init__.
    ps = getattr(args, "pipeline_strategy", None)
    if ps is None:
        ps = 0
    return WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles_wg,
        args.n_tiles_wg,
        args.k_tiles,
        2,  # placeholder, overridden by pipeline_strategy in __post_init__
        k,
        load_type=load_type,
        b_path=b_path,
        num_wg_per_cu=getattr(args, "num_wg_per_cu", 1) or 1,
        lcm_unroll=args.lcm_unroll if args.lcm_unroll is not None else True,
        unroll_factor_multiplier=getattr(args, "unroll_multiplier", 1) or 1,
        epilogue_peeling=(
            args.epilogue_peeling if args.epilogue_peeling is not None else True
        ),
        ll_sched=args.ll_sched if args.ll_sched is not None else False,
        hoist_wait=args.hoist_wait if args.hoist_wait is not None else False,
        pipeline_strategy=ps,
        _label_suffix=suffix,
    )


def _compile_fn(cfg, output_hsaco_path, **kwargs):
    """Compile wrapper -- cfg carries load_type, b_path, unroll and peeling config."""
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
        cfg_map = {c.label: c for c in to_verify}
        enriched = []
        for e in errors:
            label = e.split(":")[0].strip()
            repro = ""
            if label in cfg_map:
                try:
                    repro = f"\n  repro: {_repro_cmd(cfg_map[label], 1)}"
                except Exception:
                    pass
            enriched.append(f"{e}{repro}")
        path = _save_tmpfile("bench_verify_", enriched)
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
    parser.add_argument(
        "--k-scaling-factor",
        type=int,
        help="K scaling factor (K = factor * k_tiles * 32, each 16x32 tile = 32 K elements)",
    )
    add_single_cli_args(parser)
    parser.add_argument(
        "--use-buffer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Buffer load/store (default: sweep both on/off)",
    )
    parser.add_argument(
        "--use-flat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Flat load/store (default: sweep both on/off)",
    )
    parser.add_argument(
        "--direct-b",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="B via preshuffle/LDS bypass (default: sweep both)",
    )
    parser.add_argument(
        "--direct-a",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="A via preshuffle (default: sweep both). Implies --direct-b.",
    )
    parser.add_argument(
        "--lcm-unroll",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="LCM-based kernel loop unrolling (default: sweep both on/off)",
    )
    parser.add_argument(
        "--unroll-multiplier",
        type=int,
        default=None,
        help="Unroll factor multiplier (scales LCM unroll factor, default: 1)",
    )
    parser.add_argument(
        "--epilogue-peeling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Epilogue peeling after LCM unrolling (default: sweep both on/off)",
    )
    parser.add_argument(
        "--desired-simd-occupancy",
        type=int,
        default=None,
        help="Filter sweep to configs with this SIMD occupancy (waves per SIMD)",
    )
    parser.add_argument(
        "--ll-sched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Low-level instruction scheduler (default: sweep both on/off)",
    )
    parser.add_argument(
        "--hoist-wait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hoist iter_arg waits to loop head (default: sweep both on/off)",
    )
    parser.add_argument(
        "--pipeline-strategy",
        type=int,
        default=None,
        choices=range(0, 11),
        metavar="{0..10}",
        help="Pipeline strategy (0=none, 5+=hoist-wait guaranteed, 10=max). "
        "Default: 0 for single-config, sweep all for --sweep/--full-sweep.",
    )

    args = parser.parse_args()

    # Build load_type list from --use-buffer and --use-flat.
    # Both default to None (sweep). Pinned values filter the list.
    load_types = []
    if args.use_flat is not False:
        load_types.append("flat")
    if args.use_buffer is not False:
        load_types.append("buffer")
    if args.use_flat is True:
        load_types = [lt for lt in load_types if lt == "flat"]
    if args.use_buffer is True:
        load_types = [lt for lt in load_types if lt == "buffer"]

    # Build b_path list: derive from --direct-a / --direct-b tri-state.
    # direct_a=True implies direct_b=True.
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
    # else: all_paths stays ["lds", "direct_b", "direct_ab"] (sweep all)

    variants = [(bp, lt) for lt in load_types for bp in all_paths]
    variants = [(bp, lt) for bp, lt in variants if (bp, lt) in MLIR_FILES]

    # For single-config mode, pick first variant.
    load_type = load_types[0]
    b_path = all_paths[0]

    # TOP_K labels include suffix -- filter to selected variants.
    variant_suffixes = {_make_label_suffix(bp, lt) for bp, lt in variants}
    top_k_to_run = [
        label
        for label in _TOP_K_BASE
        if any(label.endswith(s) for s in variant_suffixes)
    ]

    if args.full_sweep or args.sweep:
        variant_str = ", ".join(f"{bp}/{lt}" for bp, lt in variants)
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
            "pipeline_strategy": "pipeline_strategy",
            "k_scaling_factor": "k_scaling_factor",
            "unroll_multiplier": "unroll_factor_multiplier",
            "desired_simd_occupancy": "simd_occupancy",
            "use_buffer": "use_buffer",
            "use_flat": "use_flat",
            "direct_b": "direct_b",
            "direct_a": "direct_a",
            "lcm_unroll": "lcm_unroll",
            "epilogue_peeling": "epilogue_peeling",
            "ll_sched": "ll_sched",
            "hoist_wait": "hoist_wait",
        }
        sweep_pins = make_sweep_pins(args, _SWEEP_ATTR_MAP)

        compile_budget = getattr(args, "compile_sample", 4096)

        all_configs = _generate_configs(
            variants,
            sample_size=compile_budget,
            check_regs=not getattr(args, "no_reg_filter", False),
            sweep_pins=sweep_pins,
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
            "k_scaling_factor",
        ]
        missing = [a for a in required if getattr(args, a) is None]
        if missing:
            flags = ", ".join(f"--{a.replace('_', '-')}" for a in missing)
            parser.error(f"Single-config mode requires: {flags}")
        run_single(
            _make_config_from_args(args, load_type, b_path),
            compile_gemm,
            args,
            execute_fn=execute_gemm_hsaco,
        )
