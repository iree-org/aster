"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM.

Sweeps tile, stage, wave, and workgroup configs with multiple iterations.
Prints incremental progress and a sorted TFLOPS table at the end.
Individual configs that fail (resource exhaustion, etc.) are skipped gracefully.

Usage (full sweep via pytest):
    pytest bench_perf_sweep_001_gemm_fp16_weak_scaled.py -s -v

Usage (single config repro via CLI):
    python bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --stages 4 --k 4096
"""

import argparse
import sys

from test_perf_001_gemm_fp16_weak_scaled import WeakScaleConfig, run_weak_scaled_gemm

# MI300X theoretical peak for f16 MFMA
MI300X_PEAK_TFLOPS_F16 = 1307.0

# Sweep grid
# Note: we delinearize 2-D along workgroups and waves. Aster does not yet support
# arbitrary integer division so we need a most-minor power of 2.
TILE_CONFIGS = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 4)]
STAGE_CONFIGS = [2, 3, 4, 5]
WAVE_CONFIGS = [(2, 2), (3, 2), (3, 4), (4, 4)]
WG_GRIDS = [(19, 16), (38, 32)]  # 304, 1216 total WGs for 304 CUs
PERF_K = [4096, 8192]
NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2

# Skip the first N active configs (after excluding KNOWN_BROKEN).
# Useful when iterating: set to the index of the last config you saw.
SKIP_FIRST_N_CONFIGS = 113

# Known-broken configs: add labels here to skip them during the sweep.
# Copy from the "Add to KNOWN_BROKEN" section printed at the end of a run.
KNOWN_BROKEN = [
    "m608xn512xk4096_wg19x16_w2x2_t1x1_s2",  # HSA_STATUS_ERROR_INVALID_ISA
    "m2736xn3072xk4096_wg19x16_w3x4_t3x3_s5",  # HSA_STATUS_ERROR_INVALID_ISA
    "m3648xn4096xk4096_wg19x16_w3x4_t4x4_s3",  # HSA_STATUS_ERROR_INVALID_ISA
]


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    return (
        f"python bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles {cfg.m_tiles} --n-tiles {cfg.n_tiles}"
        f" --stages {cfg.num_stages} --k {cfg.k}"
        f" --iterations {num_iterations}"
    )


def _print_summary_table(results, failed, skipped_labels):
    """Print sorted results table, repro commands, and failure summary."""
    if not results and not failed:
        print("\nNo configs were run.")
        return

    if results:
        results.sort(key=lambda r: r[2], reverse=True)
        hdr = f"{'#':>3} {'Config':<60} | {'Time ms':>8} | {'TFLOPS':>8} | {'% Peak':>7} | {'LDS':>6}"
        sep = "-" * len(hdr)
        print(f"\n{hdr}\n{sep}")
        for rank, (cfg, ms, tflops, pct) in enumerate(results, 1):
            lds_kb = cfg.lds_bytes / 1024
            print(
                f"{rank:>3} {cfg.label:<60} | {ms:>8.2f} | {tflops:>8.1f} | {pct:>6.1f}% | {lds_kb:>4.0f}KB"
            )

    print(
        f"\nSummary: {len(results)} passed, {len(failed)} failed"
        f", {len(skipped_labels)} excluded"
    )

    if results:
        print(f"\nRepro commands (top {min(10, len(results))}):")
        for rank, (cfg, ms, tflops, pct) in enumerate(results[:10], 1):
            print(f"  #{rank} {cfg.label}:")
            print(f"    {_repro_cmd(cfg, NUM_ITERATIONS)}")

    if failed:
        print(f"\nFailed configs ({len(failed)}):")
        for cfg, err in failed:
            first_line = err.split("\n")[0][:120]
            print(f"  {cfg.label}: {first_line}")
            print(f"    {_repro_cmd(cfg, NUM_ITERATIONS)}")

        # Print copy-pasteable exclusion list
        print("\n# Add to KNOWN_BROKEN to skip these next run:")
        for cfg, err in failed:
            first_line = err.split("\n")[0][:80]
            print(f'    "{cfg.label}",  # {first_line}')


class TestWeakScalePerf:
    """Weak-scaling TFLOPS sweep across tile/stage/wave/workgroup configs.

    Runs each config with NUM_ITERATIONS kernel launches, discards WARMUP_ITERATIONS for
    GPU frequency ramp-up, takes min of the rest. Prints incremental progress and a
    sorted table at the end.
    """

    def test_perf_sweep(self):
        results = []
        failed = []
        known_broken_set = set(KNOWN_BROKEN)

        configs = [
            WeakScaleConfig(m_wg, n_wg, m_w, n_w, m_t, n_t, stages, k)
            for k in PERF_K
            for m_wg, n_wg in WG_GRIDS
            for m_w, n_w in WAVE_CONFIGS
            for m_t, n_t in TILE_CONFIGS
            for stages in STAGE_CONFIGS
        ]

        skipped_labels = [c.label for c in configs if c.label in known_broken_set]
        active = [c for c in configs if c.label not in known_broken_set]
        active = active[SKIP_FIRST_N_CONFIGS:]

        total = len(configs)
        print(
            f"\nRunning {len(active)}/{total} configs "
            f"({len(skipped_labels)} excluded, {SKIP_FIRST_N_CONFIGS} skipped by SKIP_FIRST_N_CONFIGS)"
        )
        print(
            f"  grid: {len(PERF_K)} K x {len(WG_GRIDS)} WG x {len(WAVE_CONFIGS)} wave "
            f"x {len(TILE_CONFIGS)} tile x {len(STAGE_CONFIGS)} stage"
        )
        print(f"  iterations={NUM_ITERATIONS}, warmup={WARMUP_ITERATIONS}")
        sys.stdout.flush()

        for i, cfg in enumerate(active):
            tag = f"[{i + 1}/{len(active)}] {cfg.label}"
            print(f"\nStart sweep atom: {cfg.label}")
            print(f"  RUN   {tag}")
            print(f"        {_repro_cmd(cfg, NUM_ITERATIONS)}")
            print(f'        exclude: "{cfg.label}"')
            sys.stdout.flush()
            try:
                _, times_ns = run_weak_scaled_gemm(
                    cfg.m_wg,
                    cfg.n_wg,
                    cfg.m_waves,
                    cfg.n_waves,
                    cfg.m_tiles,
                    cfg.n_tiles,
                    cfg.num_stages,
                    cfg.k,
                    NUM_ITERATIONS,
                )
            except Exception as e:
                failed.append((cfg, str(e)))
                print(f"  FAIL  {tag}: {e}")
                sys.stdout.flush()
                continue

            measured = times_ns[WARMUP_ITERATIONS:]
            min_ns = min(measured)
            min_ms = min_ns / 1e6
            tflops = cfg.total_flops / min_ns * 1e-3
            pct_peak = tflops / MI300X_PEAK_TFLOPS_F16 * 100

            results.append((cfg, min_ms, tflops, pct_peak))
            print(
                f"  OK    {tag}: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}%)"
            )
            sys.stdout.flush()

        _print_summary_table(results, failed, skipped_labels)

        assert results, "No configs succeeded"
        best_pct = results[0][3]
        assert best_pct > 20.0, (
            f"Best config only achieved {best_pct:.1f}% of peak. "
            f"Likely a measurement error -- investigate before trusting."
        )


def _run_single(args):
    """Run a single config from CLI args."""
    cfg = WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles,
        args.n_tiles,
        args.stages,
        args.k,
    )
    print(f"Config: {cfg.label}")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(
        f"  workgroups={cfg.num_workgroups}, threads={cfg.m_waves * cfg.n_waves * 64}"
    )
    print(f"  LDS={cfg.lds_bytes} bytes ({cfg.lds_bytes / 1024:.0f} KB)")
    print(f"  iterations={args.iterations}, warmup={WARMUP_ITERATIONS}")
    sys.stdout.flush()

    _, times_ns = run_weak_scaled_gemm(
        cfg.m_wg,
        cfg.n_wg,
        cfg.m_waves,
        cfg.n_waves,
        cfg.m_tiles,
        cfg.n_tiles,
        cfg.num_stages,
        cfg.k,
        args.iterations,
    )

    measured = times_ns[WARMUP_ITERATIONS:]
    min_ns = min(measured)
    min_ms = min_ns / 1e6
    tflops = cfg.total_flops / min_ns * 1e-3
    pct_peak = tflops / MI300X_PEAK_TFLOPS_F16 * 100

    print(f"\nAll iterations (ms): {[f'{t/1e6:.2f}' for t in times_ns]}")
    print(f"Measured (post-warmup): {[f'{t/1e6:.2f}' for t in measured]}")
    print(f"Min: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}% peak)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single weak-scaled GEMM config for repro/profiling",
    )
    parser.add_argument("--m-wg", type=int, required=True, help="Workgroups along M")
    parser.add_argument("--n-wg", type=int, required=True, help="Workgroups along N")
    parser.add_argument(
        "--m-waves", type=int, required=True, help="Waves per WG along M"
    )
    parser.add_argument(
        "--n-waves", type=int, required=True, help="Waves per WG along N"
    )
    parser.add_argument(
        "--m-tiles", type=int, required=True, help="Tiles per wave along M"
    )
    parser.add_argument(
        "--n-tiles", type=int, required=True, help="Tiles per wave along N"
    )
    parser.add_argument("--stages", type=int, required=True, help="Pipeline stages")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Kernel launches (default: {NUM_ITERATIONS})",
    )
    _run_single(parser.parse_args())
