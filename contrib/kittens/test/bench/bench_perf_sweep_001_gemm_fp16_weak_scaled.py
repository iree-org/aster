"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM.

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Sequential GPU execution with subprocess isolation per config.
Individual configs that fail at either phase are skipped gracefully.

Usage (single config repro via CLI):
    python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --stages 4 --k 4096

Usage (execute a pre-compiled HSACO):
    python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --stages 4 --k 4096 \
        --hsaco /tmp/bench_hsaco/label.hsaco
"""

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep by default.
TOP_K_TO_RUN = [
    "m2432xn3072xk8192_wg38x32_w2x2_t2x3_s2",
    "m2432xn3072xk4096_wg38x32_w2x2_t2x3_s2",
    "m3648xn2048xk8192_wg38x32_w2x2_t3x2_s2",
    "m4864xn4096xk4096_wg38x32_w2x2_t4x4_s2",
    "m3648xn2048xk4096_wg38x32_w2x2_t3x2_s2",
    "m2432xn2048xk8192_wg38x32_w2x2_t2x2_s2",
    "m4864xn4096xk8192_wg38x32_w2x2_t4x4_s2",
    "m2432xn2048xk4096_wg19x16_w2x2_t4x4_s2",
    "m2432xn2048xk4096_wg38x32_w2x2_t2x2_s2",
    "m2736xn1536xk8192_wg19x16_w3x2_t3x3_s2",
]


# Known-broken configs: add labels here to skip them during the sweep.
# Copy from the "Add to KNOWN_BROKEN" section printed at the end of a run.
KNOWN_BROKEN = [
    "m1824xn512xk4096_wg19x16_w3x2_t2x1_s5",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 2 --m-tiles 2 --n-tiles 1 --stages 5 --k 4096 --iterations 5
    "m2736xn1024xk4096_wg19x16_w3x2_t3x2_s3",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 2 --m-tiles 3 --n-tiles 2 --stages 3 --k 4096 --iterations 5
    "m3648xn1024xk4096_wg38x32_w3x2_t2x1_s5",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 38 --n-wg 32 --m-waves 3 --n-waves 2 --m-tiles 2 --n-tiles 1 --stages 5 --k 4096 --iterations 5
    "m5472xn2048xk4096_wg38x32_w3x2_t3x2_s3",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 38 --n-wg 32 --m-waves 3 --n-waves 2 --m-tiles 3 --n-tiles 2 --stages 3 --k 4096 --iterations 5
    "m1824xn512xk8192_wg19x16_w3x2_t2x1_s5",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 2 --m-tiles 2 --n-tiles 1 --stages 5 --k 8192 --iterations 5
    "m2736xn1024xk8192_wg19x16_w3x2_t3x2_s3",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 2 --m-tiles 3 --n-tiles 2 --stages 3 --k 8192 --iterations 5
    "m3648xn1024xk8192_wg38x32_w3x2_t2x1_s5",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 38 --n-wg 32 --m-waves 3 --n-waves 2 --m-tiles 2 --n-tiles 1 --stages 5 --k 8192 --iterations 5
    "m5472xn2048xk8192_wg38x32_w3x2_t3x2_s3",  # compile: cannot pickle 'aster._mlir_libs._mlir.ir.DiagnosticInfo' object
    # python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --m-wg 38 --n-wg 32 --m-waves 3 --n-waves 2 --m-tiles 3 --n-tiles 2 --stages 3 --k 8192 --iterations 5
]

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Parent directory contains the test modules and kittens_helpers.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
    WeakScaleConfig,
    compile_weak_scaled_gemm,
    execute_weak_scaled_hsaco,
)

# Sentinel prefix for machine-readable result line in subprocess output.
_RESULT_SENTINEL = "BENCH_RESULT_JSON:"

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
SUBPROCESS_TIMEOUT = 120  # seconds per execution
COMPILE_WORKERS = 16  # parallel compilation processes

# Skip the first N active configs (after excluding KNOWN_BROKEN).
# Useful when iterating: set to the index of the last config you saw.
SKIP_FIRST_N_CONFIGS = 0


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    return (
        f"python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles {cfg.m_tiles} --n-tiles {cfg.n_tiles}"
        f" --stages {cfg.num_stages} --k {cfg.k}"
        f" --iterations {num_iterations}"
    )


def _exec_cmd_list(cfg, hsaco_path, num_iterations):
    """Return subprocess command list for executing a pre-compiled HSACO."""
    return [
        sys.executable,
        __file__,
        "--m-wg",
        str(cfg.m_wg),
        "--n-wg",
        str(cfg.n_wg),
        "--m-waves",
        str(cfg.m_waves),
        "--n-waves",
        str(cfg.n_waves),
        "--m-tiles",
        str(cfg.m_tiles),
        "--n-tiles",
        str(cfg.n_tiles),
        "--stages",
        str(cfg.num_stages),
        "--k",
        str(cfg.k),
        "--iterations",
        str(num_iterations),
        "--hsaco",
        hsaco_path,
    ]


def _parse_result_from_output(stdout):
    """Extract BENCH_RESULT_JSON line from subprocess stdout.

    Returns dict or None.
    """
    for line in stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    return None


def _compile_one(
    label, m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, num_stages, k, hsaco_dir
):
    """Compile a single config to HSACO.

    Called in worker process.
    """
    cfg = WeakScaleConfig(m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, num_stages, k)
    output_path = os.path.join(hsaco_dir, f"{label}.hsaco")
    compile_weak_scaled_gemm(cfg, output_path)
    return label, output_path


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


def _make_inputs(cfg):
    """Create random f16 test matrices for a given config."""
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    return A, B


def bench_perf_sweep(full_sweep=False):
    """Weak-scaling TFLOPS sweep across tile/stage/wave/workgroup configs.

    Phase 1: Parallel compilation (MLIR -> HSACO) using ProcessPoolExecutor.
    Phase 2: Sequential GPU execution in subprocesses for crash isolation.

    If TOP_K_TO_RUN is non-empty and full_sweep is False, only those labels
    are run. Otherwise runs the full grid minus KNOWN_BROKEN.
    """
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

    # Note: conservative for now, seems we get burned when flying too close to the sky.
    LDS_LIMIT = int(2**16 * 0.75)
    skipped_labels = [c.label for c in configs if c.label in known_broken_set]
    active = [
        c
        for c in configs
        if c.label not in known_broken_set and c.lds_bytes < LDS_LIMIT
    ]

    # Filter to TOP_K_TO_RUN unless empty or --full-sweep.
    if TOP_K_TO_RUN and not full_sweep:
        top_set = set(TOP_K_TO_RUN)
        active = [c for c in active if c.label in top_set]

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
    print(f"  compile_workers={COMPILE_WORKERS}")
    sys.stdout.flush()

    # -- Phase 1: Parallel compilation ---------------------------------
    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    print(
        f"\n--- Phase 1: Compiling {len(active)} configs ({COMPILE_WORKERS} workers) ---"
    )
    print(f"  hsaco_dir: {hsaco_dir}")
    sys.stdout.flush()

    hsaco_paths = {}  # label -> path
    compile_failed = {}  # label -> error string
    with ProcessPoolExecutor(max_workers=COMPILE_WORKERS) as pool:
        futures = {}
        for cfg in active:
            fut = pool.submit(
                _compile_one,
                cfg.label,
                cfg.m_wg,
                cfg.n_wg,
                cfg.m_waves,
                cfg.n_waves,
                cfg.m_tiles,
                cfg.n_tiles,
                cfg.num_stages,
                cfg.k,
                hsaco_dir,
            )
            futures[fut] = cfg

        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                label, path = fut.result()
                hsaco_paths[label] = path
                print(f"  COMPILED {cfg.label}")
            except Exception as e:
                err = str(e).split("\n")[0][:120]
                compile_failed[cfg.label] = err
                failed.append((cfg, f"compile: {err}"))
                print(f"  COMPILE_FAIL {cfg.label}: {err}")
            sys.stdout.flush()

    compiled_count = len(hsaco_paths)
    print(
        f"\nCompilation done: {compiled_count} succeeded, "
        f"{len(compile_failed)} failed"
    )
    sys.stdout.flush()

    # -- Phase 2: Sequential execution in subprocesses -----------------
    exec_active = [c for c in active if c.label in hsaco_paths]
    print(
        f"\n--- Phase 2: Executing {len(exec_active)} configs (sequential, subprocess-isolated) ---"
    )
    sys.stdout.flush()

    # cwd for subprocesses: parent of bench/ so kittens_helpers resolves.
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    for i, cfg in enumerate(exec_active):
        tag = f"[{i + 1}/{len(exec_active)}] {cfg.label}"
        hsaco_path = hsaco_paths[cfg.label]
        print(f"\nStart sweep atom: {cfg.label}")
        print(f"  RUN   {tag}")
        print(f"        {_repro_cmd(cfg, NUM_ITERATIONS)}")
        print(f'        exclude: "{cfg.label}"')
        sys.stdout.flush()

        cmd = _exec_cmd_list(cfg, hsaco_path, NUM_ITERATIONS)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                cwd=test_dir,
            )
        except subprocess.TimeoutExpired:
            failed.append((cfg, f"timeout after {SUBPROCESS_TIMEOUT}s"))
            print(f"  FAIL  {tag}: timeout after {SUBPROCESS_TIMEOUT}s")
            sys.stdout.flush()
            continue

        if proc.returncode != 0:
            stderr_lines = proc.stderr.strip().splitlines()
            err = stderr_lines[-1][:120] if stderr_lines else "unknown error"
            if proc.returncode < 0:
                import signal

                sig = -proc.returncode
                sig_name = (
                    signal.Signals(sig).name
                    if sig in signal.Signals._value2member_map_
                    else str(sig)
                )
                err = f"killed by {sig_name}: {err}"
            failed.append((cfg, err))
            print(f"  FAIL  {tag}: exit {proc.returncode}: {err}")
            sys.stdout.flush()
            continue

        result_data = _parse_result_from_output(proc.stdout)
        if result_data is None:
            failed.append((cfg, "no result line in subprocess output"))
            print(f"  FAIL  {tag}: no result line in subprocess output")
            if proc.stdout.strip():
                for line in proc.stdout.strip().splitlines()[-5:]:
                    print(f"        stdout: {line}")
            sys.stdout.flush()
            continue

        min_ms = result_data["min_ms"]
        tflops = result_data["tflops"]
        pct_peak = result_data["pct_peak"]

        results.append((cfg, min_ms, tflops, pct_peak))
        print(f"  OK    {tag}: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}%)")
        sys.stdout.flush()

    _print_summary_table(results, failed, skipped_labels)

    if not results:
        print("\nNo configs succeeded.")
        return

    print(f"\nBest {min(20, len(results))} configs:")
    for rank, (cfg, ms, tflops, pct) in enumerate(results[:20], 1):
        print(f"  #{rank} {cfg.label}: {tflops:.1f} TFLOPS ({pct:.1f}% peak)")


def _run_single(args):
    """Run a single config from CLI args.

    Emits BENCH_RESULT_JSON for sweep parsing.
    """
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
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    print(f"  LDS={cfg.lds_bytes} bytes ({cfg.lds_bytes / 1024:.0f} KB)")
    print(f"  iterations={args.iterations}, warmup={WARMUP_ITERATIONS}")
    sys.stdout.flush()

    if args.compile_only:
        if not args.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        compile_weak_scaled_gemm(cfg, args.hsaco)
        print(f"  Compiled: {args.hsaco}")
        return

    A, B = _make_inputs(cfg)

    if args.hsaco:
        # Execute pre-compiled HSACO (used by sweep phase 2 / rocprofv3).
        # Skip GPU check: if caller provided an HSACO they know the GPU is there,
        # and rocminfo hangs under rocprofv3.
        _, times_ns = execute_weak_scaled_hsaco(
            cfg, args.hsaco, args.iterations, A, B, skip_gpu_check=True
        )
    else:
        # Full compile + execute (used for standalone repro).
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            compile_weak_scaled_gemm(cfg, tmp.name)
            _, times_ns = execute_weak_scaled_hsaco(
                cfg, tmp.name, args.iterations, A, B
            )

    measured = times_ns[WARMUP_ITERATIONS:]
    min_ns = min(measured)
    min_ms = min_ns / 1e6
    tflops = cfg.total_flops / min_ns * 1e-3
    pct_peak = tflops / MI300X_PEAK_TFLOPS_F16 * 100

    print(f"\nAll iterations (ms): {[f'{t/1e6:.2f}' for t in times_ns]}")
    print(f"Measured (post-warmup): {[f'{t/1e6:.2f}' for t in measured]}")
    print(f"Min: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}% peak)")

    # Machine-readable result for sweep parent process.
    print(
        _RESULT_SENTINEL
        + json.dumps(
            {
                "min_ms": min_ms,
                "tflops": tflops,
                "pct_peak": pct_peak,
                "times_ms": [t / 1e6 for t in times_ns],
            }
        )
    )


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
    parser.add_argument(
        "--hsaco",
        type=str,
        default=None,
        help="Path to pre-compiled HSACO (skips compilation)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile HSACO and exit (requires --hsaco for output path)",
    )
    _run_single(parser.parse_args())
