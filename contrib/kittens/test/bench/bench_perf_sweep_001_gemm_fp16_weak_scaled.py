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

# Sweep grid -- 16x16 MFMA with dwordx4: 4 VGPRs per C tile (vs 16 for 32x32).
# More tiles feasible per wave, so wider multiples than 32x32 variant.
STAGE_CONFIGS = [2, 3, 4, 5]  # A pipeline depth (LDS path stages)
# B pipeline depth for direct_b (independent from A).
# b_stages = gap between b_load and b_compute + 1.
# 1 = B not pipelined; 2 = 1 stage lookahead; 3-6 = deeper lookahead.
B_STAGE_CONFIGS = [1, 2, 3, 4, 5, 6]
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

MIN_DIM = 2000  # Skip configs where M, N, or K < 3000


def _precompile_reject_reason(cfg, check_regs=True):
    """Return rejection reason string, or None if config passes pre-compile filter."""
    from aster.compiler.metadata import KernelResources

    # Pipeline depth must not exceed K-tile iterations.
    k_iters = cfg.k // (cfg.k_tiles * 32)
    if k_iters <= cfg.pipeline_depth:
        return f"K iterations ({k_iters}) <= pipeline depth ({cfg.pipeline_depth})"

    # Reject underprovisioned configs: more waves than spatial tiles wastes waves.
    if cfg.m_tiles_wg < cfg.num_waves:
        return f"m_tiles_wg={cfg.m_tiles_wg} < num_waves={cfg.num_waves}"

    # Deep B pipelining: b_stages that shifts A up causes data hazards atm.
    # Safe when pipeline_depth == a_stages (B fits within A's depth).
    # TODO: fix this in the compiler.
    if cfg.pipeline_depth > cfg.a_stages:
        return f"b_stages={cfg.b_stages} forces depth={cfg.pipeline_depth} > a_stages={cfg.a_stages}"

    est = KernelResources(
        vgpr_count=cfg.estimated_vgprs if check_regs else 0,
        agpr_count=cfg.estimated_agprs if check_regs else 0,
        lds_bytes=cfg.lds_bytes,
    )
    violations = est.check_occupancy(cfg.num_threads, num_wg_per_cu=cfg.num_wg_per_cu)
    return violations[0] if violations else None


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


def _variant_grid_info(b_path, load_type):
    """Return (dim_sizes, wave_arr, tile_arr, unroll_cfgs, b_cfgs, suffix) for a variant.

    dim_sizes is a tuple of per-axis lengths for the 8-dim cartesian product:
      (waves, occupancy, n_mult, tiles, stages, b_stages, k_factors, unroll).
    """
    import numpy as np

    suffix = _make_label_suffix(b_path, load_type)
    b_cfgs = B_STAGE_CONFIGS if b_path == "direct_b" else [0]
    unroll_cfgs = []
    for lcm in LCM_UNROLL_CONFIGS:
        for um in UNROLL_MULTIPLIERS if lcm else [1]:
            for ep in EPILOGUE_PEELING_CONFIGS:
                unroll_cfgs.append((lcm, um, ep))
    wave_arr = np.array(WAVE_CONFIGS)
    tile_arr = np.array(TILE_WG_CONFIGS)
    dim_sizes = (
        len(wave_arr),
        len(OCCUPANCY_TARGETS),
        len(N_WG_MULTIPLIERS),
        len(tile_arr),
        len(STAGE_CONFIGS),
        len(b_cfgs),
        len(K_SCALING_FACTORS),
        len(unroll_cfgs),
    )
    return dim_sizes, wave_arr, tile_arr, unroll_cfgs, b_cfgs, suffix


def _eval_batch(
    flat_indices,
    wave_arr,
    tile_arr,
    unroll_cfgs,
    b_cfgs,
    dim_sizes,
    b_path,
    load_type,
    suffix,
    check_regs,
    sweep_pins=None,
):
    """Evaluate a batch of flat grid indices. Returns list of passing WeakScaleConfigs.

    ALL filtering (dimension, occupancy, LDS, registers, sweep pins) is vectorized in
    numpy. The Python per-config loop only runs for configs that pass every check, so
    even multi-million batches are fast.
    """
    import numpy as np

    occ_arr = np.array(OCCUPANCY_TARGETS)
    nm_arr = np.array(N_WG_MULTIPLIERS)
    stg_arr = np.array(STAGE_CONFIGS)
    bstg_arr = np.array(b_cfgs)
    kf_arr = np.array(K_SCALING_FACTORS)

    # Unravel flat indices into per-dimension indices.
    multi = np.array(np.unravel_index(flat_indices, dim_sizes)).T  # (B, 8)
    iw, iocc, inm, it, ist, ib, ik, iu = multi.T

    # Look up actual values from per-dimension arrays.
    mw = wave_arr[iw, 0]
    nw = wave_arr[iw, 1]
    occ = occ_arr[iocc]
    n_mult = nm_arr[inm]
    mtwg = tile_arr[it, 0]
    ntwg = tile_arr[it, 1]
    kt = tile_arr[it, 2]
    stages = stg_arr[ist]
    b_stg = bstg_arr[ib]
    k_factor = kf_arr[ik]

    # Derived quantities (vectorized).
    num_waves = mw * nw
    waves_per_simd = (num_waves + _NUM_SIMDS - 1) // _NUM_SIMDS
    num_wg_per_cu = occ // waves_per_simd
    m_wg = _WG_BASE[0] * num_wg_per_cu
    n_wg = _WG_BASE[1] * n_mult
    k = k_factor * kt * 32
    mt = np.where(mw > 0, mtwg // np.maximum(mw, 1), 0)
    nt = np.where(nw > 0, ntwg // np.maximum(nw, 1), 0)
    eff_b_stg = np.where(b_stg > 0, b_stg, stages)

    # --- Vectorized filters (dimension + occupancy) ---
    mask = np.ones(len(flat_indices), dtype=bool)
    mask &= occ % waves_per_simd == 0
    mask &= mtwg % mw == 0
    mask &= ntwg % nw == 0
    mask &= m_wg * mtwg * 16 >= MIN_DIM
    mask &= n_wg * ntwg * 16 >= MIN_DIM
    mask &= k >= MIN_DIM
    k_iters = k // (kt * 32)
    pipeline_depth = np.maximum(stages, eff_b_stg)
    mask &= k_iters > pipeline_depth
    mask &= mtwg >= num_waves
    mask &= pipeline_depth <= stages

    # --- Vectorized sweep pin filter ---
    if sweep_pins:
        # Compute derived quantities needed for some pins.
        simd_occ = num_wg_per_cu * ((num_waves + _NUM_SIMDS - 1) // _NUM_SIMDS)
        # unroll_factor_multiplier is per-config from unroll_cfgs.
        um_vals = np.array([unroll_cfgs[i][1] for i in iu])
        pin_arrays = {
            "m_wg": m_wg,
            "n_wg": n_wg,
            "m_waves": mw,
            "n_waves": nw,
            "m_tiles_wg": mtwg,
            "n_tiles_wg": ntwg,
            "k_tiles": kt,
            "a_stages": stages,
            "k_scaling_factor": k_factor,
            "simd_occupancy": simd_occ,
            "unroll_factor_multiplier": um_vals,
        }
        for attr, val in sweep_pins.items():
            if attr in pin_arrays:
                mask &= pin_arrays[attr] == val

    # --- Vectorized LDS + register checks (replaces _precompile_reject_reason) ---
    if check_regs:
        # LDS: a_stages * m_tiles_wg * k_tiles * 1024 + eff_b_stages * n_tiles_wg * k_tiles * 1024
        is_direct_b = b_path in ("direct_b", "direct_ab")
        a_lds = stages * mtwg * kt * 1024
        b_lds = np.int64(0) if is_direct_b else eff_b_stg * ntwg * kt * 1024
        lds_bytes = a_lds + b_lds
        lds_budget = np.int64(65536) // np.maximum(num_wg_per_cu, 1)
        mask &= lds_bytes <= lds_budget

        # VGPR estimate (vectorized version of estimated_vgprs).
        # Cooperative split for A: coop_m * coop_k
        waves_m = np.minimum(mtwg, num_waves)
        waves_k_a = np.maximum(1, num_waves // np.maximum(waves_m, 1))
        coop_m = (mtwg + waves_m - 1) // np.maximum(waves_m, 1)
        coop_k_a = (kt + waves_k_a - 1) // np.maximum(waves_k_a, 1)
        coop_a_mk = coop_m * coop_k_a

        a_load_bufs = coop_a_mk * stages * 4
        a_lds_read = mt * kt * 4
        if is_direct_b:
            b_load_bufs = nt * kt * eff_b_stg * 4
            b_split = nt * kt * 4
            overhead = 30
        else:
            waves_n = np.minimum(ntwg, num_waves)
            waves_k_b = np.maximum(1, num_waves // np.maximum(waves_n, 1))
            coop_n = (ntwg + waves_n - 1) // np.maximum(waves_n, 1)
            coop_k_b = (kt + waves_k_b - 1) // np.maximum(waves_k_b, 1)
            coop_b_nk = coop_n * coop_k_b
            b_load_bufs = coop_b_nk * stages * 4
            b_split = nt * kt * 4
            overhead = 10

        est_vgprs = a_load_bufs + a_lds_read + b_load_bufs + b_split + overhead
        est_agprs = mt * nt * 4

        # Per-wave limits (ISA manual: 256 VGPRs, 256 AGPRs, 512 combined).
        mask &= est_vgprs <= 256
        mask &= est_agprs <= 256
        mask &= (est_vgprs + est_agprs) <= 512

    # --- Instantiate only fully-passing configs (fast: small survivor set) ---
    configs = []
    for i in np.where(mask)[0]:
        lcm_val, um_val, ep_val = unroll_cfgs[iu[i]]
        cfg = WeakScaleConfig(
            int(m_wg[i]),
            int(n_wg[i]),
            int(mw[i]),
            int(nw[i]),
            int(mtwg[i]),
            int(ntwg[i]),
            int(kt[i]),
            int(stages[i]),
            int(k[i]),
            load_type=load_type,
            b_path=b_path,
            b_stages=int(b_stg[i]),
            num_wg_per_cu=int(num_wg_per_cu[i]),
            lcm_unroll=bool(lcm_val),
            unroll_factor_multiplier=int(um_val),
            epilogue_peeling=bool(ep_val),
            _label_suffix=suffix,
        )
        configs.append(cfg)
    return configs, int(np.sum(~mask))


def _generate_configs(
    variants=None, sample_size=3000, check_regs=True, sweep_pins=None
):
    """Generate sweep configs by sampling the grid -- never materializes the full product.

    The cartesian product across 8 dimensions can be 38M+ rows per variant.
    Instead of building it all in memory, we:
      1. Compute the total grid size per variant (just a product of dim lengths).
      2. Sample random flat indices into the grid.
      3. Unravel + compute + filter only the sampled rows (ALL filters vectorized).
      4. Repeat in rounds if the acceptance rate is low.

    sweep_pins is a dict {config_attr: value} applied vectorized in numpy.
    O(sample_size) memory, not O(grid_size).
    """
    import numpy as np

    if variants is None:
        variants = list(MLIR_FILES.keys())

    rng = np.random.default_rng()
    configs = []
    total_filtered = 0
    total_sampled = 0
    total_grid = 0

    for b_path, load_type in variants:
        if (b_path, load_type) not in MLIR_FILES:
            continue
        dim_sizes, wave_arr, tile_arr, unroll_cfgs, b_cfgs, suffix = _variant_grid_info(
            b_path, load_type
        )
        grid_total = int(np.prod(dim_sizes))
        total_grid += grid_total

        eval_args = (
            wave_arr,
            tile_arr,
            unroll_cfgs,
            b_cfgs,
            dim_sizes,
            b_path,
            load_type,
            suffix,
            check_regs,
            sweep_pins,
        )

        if sample_size <= 0:
            # Full sweep: evaluate entire grid in chunks to bound memory.
            chunk_size = 500_000
            for start in range(0, grid_total, chunk_size):
                end = min(start + chunk_size, grid_total)
                batch_cfgs, nfilt = _eval_batch(np.arange(start, end), *eval_args)
                configs.extend(batch_cfgs)
                total_filtered += nfilt
                total_sampled += end - start
        else:
            # Sampled sweep: draw random indices in rounds.
            # All filters (including sweep_pins) are vectorized in _eval_batch,
            # so target means post-all-filter count.
            target = sample_size
            seen = set()
            batch_size = min(target * 10, grid_total)
            max_rounds = 50
            accept_count = 0
            eval_count = 0
            for round_idx in range(max_rounds):
                draw_size = min(batch_size, grid_total - len(seen))
                if draw_size <= 0:
                    break
                candidates = rng.choice(grid_total, size=draw_size, replace=False)
                new = np.array([c for c in candidates if c not in seen])
                if len(new) == 0:
                    break
                seen.update(new.tolist())
                batch_cfgs, nfilt = _eval_batch(new, *eval_args)
                total_filtered += nfilt
                total_sampled += len(new)
                eval_count += len(new)
                accept_count += len(batch_cfgs)
                configs.extend(batch_cfgs)
                if len(configs) >= target:
                    break
                # Adapt batch size based on observed acceptance rate.
                if accept_count > 0 and round_idx >= 1:
                    rate = accept_count / eval_count
                    needed = target - len(configs)
                    batch_size = min(int(needed / rate * 3) + 1000, grid_total)

    if total_filtered:
        print(f"{total_filtered} configs skipped by pre-compile filter")

    total = len(configs)
    # When sampling, the reported eligible count is only from sampled grid
    # points. Extrapolate to give the user a sense of the full grid.
    if sample_size > 0 and total_sampled > 0 and total_sampled < total_grid:
        accept_rate = total / total_sampled
        est_eligible = int(accept_rate * total_grid)
        print(
            f"Grid: {total_grid} total, sampled {total_sampled}, "
            f"found {total} eligible (~{accept_rate:.1%} accept rate, "
            f"~{est_eligible} estimated eligible in full grid)"
        )
    n = min(sample_size, total) if sample_size > 0 else total
    if n < total:
        configs = _stratified_sample(configs, n)
    shapes = {
        (c.m_waves, c.n_waves, c.m_tiles_wg, c.n_tiles_wg, c.k_tiles) for c in configs
    }
    if n < total:
        print(
            f"Compiling {len(configs)} / {total} eligible configs ({len(shapes)} distinct shapes)"
        )
    else:
        print(f"Compiling all {total} eligible configs ({len(shapes)} distinct shapes)")
    return configs


def _stratified_sample(configs, n):
    """Sample uniformly across structural kernel shapes, then within each.

    The cartesian grid produces many micro-variants (stages, k_factor, unroll) per
    distinct kernel shape (waves, tiles, k_tiles, path). Flat random.sample over-
    represents shapes that have many surviving micro-variants. Stratified sampling
    ensures diverse shape coverage and returns exactly n configs (or all configs if
    fewer than n exist).
    """
    import random
    from collections import defaultdict

    buckets = defaultdict(list)
    for cfg in configs:
        key = (
            cfg.m_waves,
            cfg.n_waves,
            cfg.m_tiles_wg,
            cfg.n_tiles_wg,
            cfg.k_tiles,
            cfg.b_path,
            cfg.load_type,
        )
        buckets[key].append(cfg)

    num_shapes = len(buckets)
    if num_shapes == 0:
        return []

    # Shuffle groups internally so we pick different micro-variants each run.
    for group in buckets.values():
        random.shuffle(group)

    # Round-robin allocation: give each shape a fair base quota, then
    # redistribute any unfilled slots (small shapes) to larger shapes.
    remaining = n
    quotas = {}
    base = remaining // num_shapes
    extra = remaining - base * num_shapes
    keys = list(buckets.keys())
    random.shuffle(keys)
    for i, key in enumerate(keys):
        quotas[key] = base + (1 if i < extra else 0)

    # First pass: fill quotas, track underfilled shapes.
    result = []
    surplus = 0
    underfilled = []
    for key in keys:
        group = buckets[key]
        quota = quotas[key]
        take = min(quota, len(group))
        result.extend(group[:take])
        if take < quota:
            surplus += quota - take
        else:
            underfilled.append(key)

    # Second pass: distribute surplus to shapes that have leftover configs.
    if surplus > 0:
        available = [
            (key, len(buckets[key]) - quotas[key])
            for key in keys
            if len(buckets[key]) > quotas[key]
        ]
        random.shuffle(available)
        for key, headroom in available:
            give = min(surplus, headroom)
            if give <= 0:
                continue
            already = quotas[key]
            result.extend(buckets[key][already : already + give])
            surplus -= give
            if surplus <= 0:
                break

    random.shuffle(result)
    return result


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 32)
    buf_flag = " --use-buffer" if cfg.use_buffer else " --use-flat"
    if cfg.direct_a:
        direct_flag = " --direct-a --direct-b"
    elif cfg.direct_b:
        direct_flag = " --direct-b"
    else:
        direct_flag = ""
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
    ll_flag = " --ll-sched" if getattr(cfg, "ll_sched", False) else ""
    return (
        f"python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles-wg {cfg.m_tiles_wg} --n-tiles-wg {cfg.n_tiles_wg} --k-tiles {cfg.k_tiles}"
        f" --a-stages {cfg.a_stages} --k-scaling-factor {k_factor}"
        f"{f' --b-stages {cfg.b_stages}' if cfg.b_stages > 0 else ''}"
        f"{buf_flag}{direct_flag}{lcm_flag}{um_flag}{peel_flag}{wg_per_cu_flag}"
        f"{ll_flag}"
        f" --iterations {num_iterations}"
    )


def _make_config_from_args(args, load_type, b_path):
    """Construct a WeakScaleConfig from parsed CLI args."""
    k = args.k_scaling_factor * args.k_tiles * 32
    suffix = _make_label_suffix(b_path, load_type)
    stages = args.a_stages
    return WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles_wg,
        args.n_tiles_wg,
        args.k_tiles,
        stages,
        k,
        load_type=load_type,
        b_path=b_path,
        b_stages=getattr(args, "b_stages", 0) or 0,
        num_wg_per_cu=getattr(args, "num_wg_per_cu", 1) or 1,
        lcm_unroll=getattr(args, "lcm_unroll", True),
        unroll_factor_multiplier=getattr(args, "unroll_multiplier", 1) or 1,
        epilogue_peeling=getattr(args, "epilogue_peeling", True),
        ll_sched=getattr(args, "ll_sched", False),
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
    parser.add_argument("--a-stages", type=int, help="A pipeline stages (LDS path)")
    parser.add_argument(
        "--b-stages",
        type=int,
        default=0,
        help="B pipeline stages for direct_b (0=same as A, 1=none, 2=load overlap, 3=full)",
    )
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
        help="A via preshuffle (LDS bypass). Requires --direct-b.",
    )
    parser.add_argument(
        "--direct-b",
        action="store_true",
        help="B via preshuffle (LDS bypass).",
    )
    parser.add_argument(
        "--no-direct-a",
        action="store_true",
        help="Exclude direct_ab from sweep (keep lds + direct_b).",
    )
    parser.add_argument(
        "--no-direct-b",
        action="store_true",
        help="Exclude direct_b and direct_ab from sweep (keep lds only).",
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
    parser.add_argument(
        "--ll-sched",
        action="store_true",
        help="Enable low-level instruction scheduler (off by default)",
    )

    args = parser.parse_args()
    args.lcm_unroll = not args.no_lcm_unroll
    args.epilogue_peeling = not args.no_epilogue_peeling

    # Determine b_path from --direct-a / --direct-b flags.
    if args.direct_a and not args.direct_b:
        parser.error("--direct-a requires --direct-b")
    if args.direct_a and args.direct_b:
        b_path = "direct_ab"
    elif args.direct_b:
        b_path = "direct_b"
    else:
        b_path = "lds"

    # Determine load_type variants to sweep.
    if args.use_buffer:
        load_types = ["buffer"]
    elif args.use_flat:
        load_types = ["flat"]
    else:
        load_types = ["flat", "buffer"]

    # Build (b_path, load_type) variant list.
    if args.full_sweep or args.sweep:
        if args.direct_a or args.direct_b:
            # Pinned to the specified path.
            variants = [(b_path, lt) for lt in load_types]
        else:
            # Sweep paths, respecting --no-direct-a / --no-direct-b exclusions.
            all_paths = ["lds", "direct_b", "direct_ab"]
            if args.no_direct_b:
                all_paths = ["lds"]
            elif args.no_direct_a:
                all_paths = ["lds", "direct_b"]
            variants = [(bp, lt) for lt in load_types for bp in all_paths]
    else:
        variants = [(b_path, lt) for lt in load_types]

    # Filter to implemented combos
    variants = [(bp, lt) for bp, lt in variants if (bp, lt) in MLIR_FILES]

    # For single-config mode
    load_type = "buffer" if args.use_buffer else "flat"

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
            "a_stages": "a_stages",
            "k_scaling_factor": "k_scaling_factor",
            "unroll_multiplier": "unroll_factor_multiplier",
            "desired_simd_occupancy": "simd_occupancy",
        }
        sweep_pins = make_sweep_pins(args, _SWEEP_ATTR_MAP)

        all_configs = _generate_configs(
            variants,
            sample_size=getattr(args, "compile_sample", 4096),
            check_regs=not getattr(args, "no_reg_filter", False),
            sweep_pins=sweep_pins,
        )
        # Propagate pipeline flags to all generated configs.
        for cfg in all_configs:
            cfg.ll_sched = args.ll_sched

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
            "a_stages",
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
