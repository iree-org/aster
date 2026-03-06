"""Benchmark: Weak-scaling TFLOPS sweep for constexpr GEMM.

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.
Individual configs that fail at either phase are skipped gracefully.

Usage (partial sweep / full sweep):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --full-sweep
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config compile + run):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --k-tiles 1 --stages 4 --k-scaling-factor 256

Usage (compile only, produce HSACO):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --k-tiles 1 --stages 4 --k-scaling-factor 256 \
        --compile-only --hsaco /tmp/output.hsaco

Usage (execute a pre-compiled HSACO):
    python contrib/kittens/test/bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py \
        --m-wg 19 --n-wg 16 --m-waves 3 --n-waves 4 \
        --m-tiles 2 --n-tiles 3 --k-tiles 1 --stages 4 --k-scaling-factor 256 \
        --hsaco /tmp/output.hsaco
"""

# IMPORTANT: Top configs to run by default. If non-empty, only these labels are run
# unless --full-sweep is passed. Empty list = full sweep by default.
TOP_K_TO_RUN = [  
    "m4864xn4096xk8192_wg38x32_w2x2_t4x4x2_s2",
    "m4864xn4096xk4096_wg38x32_w2x2_t4x4x2_s2",
    "m3648xn4096xk4096_wg38x32_w2x2_t3x4x2_s2",
    "m3648xn4096xk8192_wg38x32_w2x2_t3x4x2_s2",
    "m2432xn4096xk8192_wg38x32_w2x2_t2x4x2_s2",
    "m4864xn2048xk8192_wg38x32_w2x2_t4x2x2_s2",
    "m3648xn2048xk8192_wg19x16_w3x2_t4x4x2_s2",
    "m2432xn4096xk4096_wg38x32_w2x2_t2x4x2_s2",
    "m4864xn2048xk4096_wg38x32_w2x2_t4x2x2_s2",
    "m3648xn2048xk4096_wg38x32_w2x2_t3x2x2_s3",
]


# Known-broken configs: add labels here to skip them during the sweep.
# Copy from the "Add to KNOWN_BROKEN" section printed at the end of a run.
KNOWN_BROKEN = [
    # Populate after a full sweep with the new K = factor * k_tiles * 16 grid.
]

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

# Parent directory contains the test modules and kittens_helpers.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Worktree root contains mlir_kernels (used by library preloading).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
    KERNEL_NAME,
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
TILE_CONFIGS = [
    (2, 2, 2),
    (2, 4, 2),
    (2, 2, 4),
    (2, 4, 4),
    (3, 2, 2),
    (3, 4, 2),
    (3, 2, 4),
    (3, 4, 4),
    (4, 2, 2),
    (4, 4, 2),
    (4, 2, 4),
    (4, 4, 4),
]
STAGE_CONFIGS = [2, 3, 4, 5]
WAVE_CONFIGS = [(2, 2), (3, 2), (3, 4), (4, 4)]
WG_GRIDS = [(19, 16), (38, 32)]  # 304, 1216 total WGs for 304 CUs
K_SCALING_FACTORS = [128, 256]  # K = factor * k_tiles * 16
NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2
SUBPROCESS_TIMEOUT = 120  # seconds per execution
COMPILE_WORKERS = 8  # parallel compilation processes

# Skip the first N active configs (after excluding KNOWN_BROKEN).
# Useful when iterating: set to the index of the last config you saw.
SKIP_FIRST_N_CONFIGS = 0


def _detect_num_gpus():
    """Detect number of available GPUs via HIP.

    Returns 1 if detection fails (safe fallback to sequential execution).
    """
    try:
        from aster.testing import hip_get_device_count

        return max(1, hip_get_device_count())
    except Exception:
        return 1


def _repro_cmd(cfg, num_iterations):
    """Return a CLI command to reproduce a single config."""
    k_factor = cfg.k // (cfg.k_tiles * 16)
    return (
        f"python bench/bench_perf_sweep_001_gemm_fp16_weak_scaled.py"
        f" --m-wg {cfg.m_wg} --n-wg {cfg.n_wg}"
        f" --m-waves {cfg.m_waves} --n-waves {cfg.n_waves}"
        f" --m-tiles {cfg.m_tiles} --n-tiles {cfg.n_tiles} --k-tiles {cfg.k_tiles}"
        f" --stages {cfg.num_stages} --k-scaling-factor {k_factor}"
        f" --iterations {num_iterations}"
    )


def _exec_cmd_list(cfg, hsaco_path, num_iterations):
    """Return subprocess command list for executing a pre-compiled HSACO."""
    k_factor = cfg.k // (cfg.k_tiles * 16)
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
        "--k-tiles",
        str(cfg.k_tiles),
        "--k-scaling-factor",
        str(k_factor),
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


def _format_mlir_error(e):
    """Extract actionable diagnostics from an MLIRError.

    MLIR pass failures raise ir.MLIRError whose str() is just "MLIRError". The actual
    diagnostics (LDS allocation failure, regalloc failure, etc.) are in
    e.error_diagnostics[].message and .notes[].message.
    """
    parts = []
    diagnostics = getattr(e, "error_diagnostics", None)
    if diagnostics:
        for diag in diagnostics:
            msg = getattr(diag, "message", "")
            if msg:
                parts.append(msg)
            for note in getattr(diag, "notes", []):
                note_msg = getattr(note, "message", "")
                if note_msg:
                    parts.append(f"  note: {note_msg}")
    if parts:
        return "\n".join(parts)
    # Fallback: use full str(e) which may at least have the exception type.
    return str(e) or type(e).__name__


def _compile_one(
    label,
    m_wg,
    n_wg,
    m_waves,
    n_waves,
    m_tiles,
    n_tiles,
    k_tiles,
    num_stages,
    k,
    hsaco_dir,
):
    """Compile a single config to HSACO.

    Called in worker process. Returns (label, hsaco_path, KernelResources|None). Raises
    RuntimeError with actionable diagnostics on failure.
    """
    from aster.hip import parse_asm_kernel_resources

    cfg = WeakScaleConfig(
        m_wg,
        n_wg,
        m_waves,
        n_waves,
        m_tiles,
        n_tiles,
        k_tiles,
        num_stages,
        k,
    )
    output_path = os.path.join(hsaco_dir, f"{label}.hsaco")
    try:
        _, asm = compile_weak_scaled_gemm(cfg, output_path)
    except Exception as e:
        # Re-raise with actionable diagnostics extracted from MLIRError.
        raise RuntimeError(_format_mlir_error(e)) from None
    # Save ASM alongside HSACO for later inspection / resource parsing.
    asm_path = output_path.replace(".hsaco", ".s")
    with open(asm_path, "w") as f:
        f.write(asm)
    res = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME).get(KERNEL_NAME)
    return label, output_path, res


def _print_summary_table(results, failed, skipped_labels, resources_map=None):
    """Print sorted results table, repro commands, and failure summary."""
    if resources_map is None:
        resources_map = {}
    if not results and not failed:
        print("\nNo configs were run.")
        return

    if results:
        results.sort(key=lambda r: r[2], reverse=True)
        hdr = (
            f"{'#':>3} {'Config':<60} | {'Time ms':>8} | {'TFLOPS':>8} "
            f"| {'% Peak':>7} | {'LDS':>6} | {'Resources'}"
        )
        sep = "-" * len(hdr)
        print(f"\n{hdr}\n{sep}")
        for rank, (cfg, ms, tflops, pct) in enumerate(results, 1):
            res = resources_map.get(cfg.label)
            lds_kb = (res.lds_bytes if res else cfg.lds_bytes) / 1024
            res_str = res.registers_str if res else ""
            print(
                f"{rank:>3} {cfg.label:<60} | {ms:>8.2f} | {tflops:>8.1f} "
                f"| {pct:>6.1f}% | {lds_kb:>4.0f}KB | {res_str}"
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
        # Categorize failures for actionable summary.
        categories = {}
        for cfg, err in failed:
            # Classify by first diagnostic line.
            first_line = err.split("\n")[0]
            if "failed to allocate LDS" in err or "LDS" in first_line:
                cat = "LDS_ALLOC"
            elif (
                "failed to run register allocator" in err
                or "register" in first_line.lower()
            ):
                cat = "REGALLOC"
            elif "compile:" in err:
                cat = "COMPILE"
            else:
                cat = "RUNTIME"
            categories.setdefault(cat, []).append((cfg, err))

        print(f"\nFailed configs ({len(failed)}):")
        for cat, items in sorted(categories.items()):
            print(f"\n  [{cat}] ({len(items)} configs):")
            for cfg, err in items:
                first_line = err.split("\n")[0][:200]
                print(f"    {cfg.label}: {first_line}")
                # Show additional diagnostic lines (notes) indented.
                for extra_line in err.split("\n")[1:3]:
                    if extra_line.strip():
                        print(f"      {extra_line.strip()}")
                print(f"      repro: {_repro_cmd(cfg, NUM_ITERATIONS)}")

        # Print copy-pasteable exclusion list
        print("\n# Add to KNOWN_BROKEN to skip these next run:")
        for cfg, err in failed:
            first_line = err.split("\n")[0][:80]
            print(f'    "{cfg.label}",  # {first_line}')


def _exec_one_config(cfg, hsaco_path, num_iterations, test_dir, gpu_id):
    """Execute a single config in a subprocess pinned to a specific GPU.

    Returns (cfg, result_data, error_string). On success error_string is None; on
    failure result_data is None.
    """
    cmd = _exec_cmd_list(cfg, hsaco_path, num_iterations)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            cwd=test_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return cfg, None, f"timeout after {SUBPROCESS_TIMEOUT}s"

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
        return cfg, None, f"exit {proc.returncode}: {err}"

    result_data = _parse_result_from_output(proc.stdout)
    if result_data is None:
        tail = ""
        if proc.stdout.strip():
            tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
        return cfg, None, f"no result line in subprocess output\n{tail}"

    return cfg, result_data, None


def _make_inputs(cfg):
    """Create random f16 test matrices for a given config."""
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    return A, B


def bench_perf_sweep(full_sweep=False, num_gpus=None, compile_workers=None):
    """Weak-scaling TFLOPS sweep across tile/stage/wave/workgroup configs.

    Phase 1: Parallel compilation (MLIR -> HSACO) using ProcessPoolExecutor.
    Phase 2: Parallel GPU execution across available GPUs (round-robin),
             each config in its own subprocess for crash isolation.

    If TOP_K_TO_RUN is non-empty and full_sweep is False, only those labels
    are run. Otherwise runs the full grid minus KNOWN_BROKEN.
    """
    if num_gpus is None:
        num_gpus = _detect_num_gpus()
    if compile_workers is None:
        compile_workers = COMPILE_WORKERS
    results = []
    failed = []
    known_broken_set = set(KNOWN_BROKEN)

    configs = [
        WeakScaleConfig(
            m_wg, n_wg, m_w, n_w, m_t, n_t, k_t, stages, k_factor * k_t * 16
        )
        for k_factor in K_SCALING_FACTORS
        for m_wg, n_wg in WG_GRIDS
        for m_w, n_w in WAVE_CONFIGS
        for m_t, n_t, k_t in TILE_CONFIGS
        for stages in STAGE_CONFIGS
    ]

    skipped_labels = [c.label for c in configs if c.label in known_broken_set]
    active = [c for c in configs if c.label not in known_broken_set]

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
        f"  grid: {len(K_SCALING_FACTORS)} K_factor x {len(WG_GRIDS)} WG x {len(WAVE_CONFIGS)} wave "
        f"x {len(TILE_CONFIGS)} tile x {len(STAGE_CONFIGS)} stage"
    )
    print(f"  iterations={NUM_ITERATIONS}, warmup={WARMUP_ITERATIONS}")
    print(f"  compile_workers={compile_workers}, exec_gpus={num_gpus}")
    sys.stdout.flush()

    # -- Phase 1: Parallel compilation ---------------------------------
    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    print(
        f"\n--- Phase 1: Compiling {len(active)} configs ({compile_workers} workers) ---"
    )
    print(f"  hsaco_dir: {hsaco_dir}")
    sys.stdout.flush()

    hsaco_paths = {}  # label -> path
    resources_map = {}  # label -> KernelResources
    compile_failed = {}  # label -> error string
    with ProcessPoolExecutor(max_workers=compile_workers) as pool:
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
                cfg.k_tiles,
                cfg.num_stages,
                cfg.k,
                hsaco_dir,
            )
            futures[fut] = cfg

        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                label, path, res = fut.result()
                hsaco_paths[label] = path
                if res:
                    resources_map[label] = res
                print(f"  COMPILED {cfg.label}  [{res or '?'}]")
            except Exception as e:
                err = str(e)
                first_line = err.split("\n")[0][:200]
                compile_failed[cfg.label] = err
                failed.append((cfg, f"compile: {err}"))
                print(f"  COMPILE_FAIL {cfg.label}: {first_line}")
            sys.stdout.flush()

    compiled_count = len(hsaco_paths)
    print(
        f"\nCompilation done: {compiled_count} succeeded, "
        f"{len(compile_failed)} failed"
    )
    sys.stdout.flush()

    # -- Phase 2: Parallel execution across GPUs -------------------------
    exec_active = [c for c in active if c.label in hsaco_paths]
    print(
        f"\n--- Phase 2: Executing {len(exec_active)} configs "
        f"({num_gpus} GPU(s), subprocess-isolated) ---"
    )
    sys.stdout.flush()

    # cwd for subprocesses: parent of bench/ so kittens_helpers resolves.
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    # Use threads to manage concurrent subprocesses (one per GPU, round-robin).
    # Each subprocess is pinned to a GPU via HIP_VISIBLE_DEVICES.
    with ThreadPoolExecutor(max_workers=num_gpus) as exec_pool:
        future_to_info = {}
        for i, cfg in enumerate(exec_active):
            gpu_id = i % num_gpus
            hsaco_path = hsaco_paths[cfg.label]
            res = resources_map.get(cfg.label)
            tag = f"[{i + 1}/{len(exec_active)}] {cfg.label}"
            print(f"  SUBMIT {tag} -> GPU {gpu_id}  [{res or '?'}]")
            sys.stdout.flush()
            fut = exec_pool.submit(
                _exec_one_config,
                cfg,
                hsaco_path,
                NUM_ITERATIONS,
                test_dir,
                gpu_id,
            )
            future_to_info[fut] = (i, tag)

        for fut in as_completed(future_to_info):
            idx, tag = future_to_info[fut]
            cfg, result_data, err = fut.result()
            if err is not None:
                failed.append((cfg, err))
                print(f"  FAIL  {tag}: {err.splitlines()[0]}")
                sys.stdout.flush()
                continue

            min_ms = result_data["min_ms"]
            tflops = result_data["tflops"]
            pct_peak = result_data["pct_peak"]

            results.append((cfg, min_ms, tflops, pct_peak))
            print(
                f"  OK    {tag}: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}%)"
            )
            sys.stdout.flush()

    _print_summary_table(results, failed, skipped_labels, resources_map)

    if not results:
        print("\nNo configs succeeded.")
        return

    print(f"\nBest {min(20, len(results))} configs:")
    for rank, (cfg, ms, tflops, pct) in enumerate(results[:20], 1):
        print(f"  #{rank} {cfg.label}: {tflops:.1f} TFLOPS ({pct:.1f}% peak)")


def _print_config(cfg, iterations, resources=None):
    """Print config summary.

    Called after compilation so ASM-level resource usage is available. resources is a
    KernelResources object (or None if unavailable).
    """
    print(f"Config: {cfg.label}")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    if resources:
        print(
            f"  LDS={resources.lds_bytes} bytes ({resources.lds_bytes / 1024:.0f} KB)"
        )
        print(f"  registers: {resources.registers_str}")
        if resources.scratch_bytes > 0:
            print(f"  scratch={resources.scratch_bytes} bytes")
    print(f"  iterations={iterations}, warmup={WARMUP_ITERATIONS}")
    sys.stdout.flush()


def _run_single(args):
    """Run a single config from CLI args.

    Emits BENCH_RESULT_JSON for sweep parsing.
    """
    from aster.hip import parse_asm_kernel_resources

    k = args.k_scaling_factor * args.k_tiles * 16
    cfg = WeakScaleConfig(
        args.m_wg,
        args.n_wg,
        args.m_waves,
        args.n_waves,
        args.m_tiles,
        args.n_tiles,
        args.k_tiles,
        args.stages,
        k,
    )

    if args.compile_only:
        if not args.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        _, asm = compile_weak_scaled_gemm(cfg, args.hsaco)
        resources = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME)
        _print_config(cfg, args.iterations, resources.get(KERNEL_NAME))
        print(f"  Compiled: {args.hsaco}")
        return

    A, B = _make_inputs(cfg)

    if args.hsaco:
        # Execute pre-compiled HSACO (used by sweep phase 2 / rocprofv3).
        # Read companion .s file if available (written at compile time).
        asm_path = args.hsaco.replace(".hsaco", ".s")
        res = None
        if os.path.exists(asm_path):
            with open(asm_path) as f:
                res = parse_asm_kernel_resources(f.read(), kernel_name=KERNEL_NAME).get(
                    KERNEL_NAME
                )
        _print_config(cfg, args.iterations, res)
        _, times_ns = execute_weak_scaled_hsaco(
            cfg, args.hsaco, args.iterations, A, B, skip_gpu_check=True
        )
    else:
        # Full compile + execute (used for standalone repro).
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            _, asm = compile_weak_scaled_gemm(cfg, tmp.name)
            resources = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME)
            _print_config(cfg, args.iterations, resources.get(KERNEL_NAME))
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
        description="Weak-scaled GEMM benchmark: sweep or single-config repro",
    )
    # Sweep mode
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the benchmark sweep (TOP_K_TO_RUN by default)",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run all configs in the sweep grid (implies --sweep)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel execution (default: auto-detect)",
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=COMPILE_WORKERS,
        help=f"Parallel compilation processes (default: {COMPILE_WORKERS})",
    )
    # Single-config args (required unless --sweep)
    parser.add_argument("--m-wg", type=int, help="Workgroups along M")
    parser.add_argument("--n-wg", type=int, help="Workgroups along N")
    parser.add_argument("--m-waves", type=int, help="Waves per WG along M")
    parser.add_argument("--n-waves", type=int, help="Waves per WG along N")
    parser.add_argument("--m-tiles", type=int, help="Tiles per wave along M")
    parser.add_argument("--n-tiles", type=int, help="Tiles per wave along N")
    parser.add_argument("--k-tiles", type=int, help="Tiles per wave along K")
    parser.add_argument("--stages", type=int, help="Pipeline stages")
    parser.add_argument(
        "--k-scaling-factor",
        type=int,
        help="K scaling factor (K = factor * k_tiles * 16)",
    )
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

    args = parser.parse_args()
    if args.full_sweep or args.sweep:
        bench_perf_sweep(
            full_sweep=args.full_sweep,
            num_gpus=args.num_gpus,
            compile_workers=args.compile_workers,
        )
    else:
        # Validate required args for single-config mode.
        required = [
            "m_wg",
            "n_wg",
            "m_waves",
            "n_waves",
            "m_tiles",
            "n_tiles",
            "k_tiles",
            "stages",
            "k_scaling_factor",
        ]
        missing = [a for a in required if getattr(args, a) is None]
        if missing:
            flags = ", ".join(f"--{a.replace('_', '-')}" for a in missing)
            parser.error(f"Single-config mode requires: {flags}")
        _run_single(args)
