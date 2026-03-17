"""Shared benchmark harness for weak-scaled GEMM sweeps.

Provides the full Phase 1 (parallel compilation) + Phase 2 (parallel GPU execution)
pipeline, parameterized by config-specific callbacks. Each bench_perf_sweep_NNN script
is a thin wrapper that supplies:
  - Sweep grid (configs list)
  - compile_fn(cfg, output_path) -> (path, asm)
  - cfg_to_cli_args(cfg) -> list[str]  (for subprocess invocation)
  - repro_cmd(cfg) -> str
  - make_config_from_args(args) -> cfg  (for single-config CLI)
  - CLI arg setup (add_single_config_args)
"""

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

# Sentinel prefix for machine-readable result line in subprocess output.
RESULT_SENTINEL = "BENCH_RESULT_JSON:"

# MI300X theoretical peak for f16 MFMA
MI300X_PEAK_TFLOPS_F16 = 1307.0

# Defaults
NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2
SUBPROCESS_TIMEOUT = 120  # seconds per execution
DEFAULT_COMPILE_WORKERS = 8


def check_numpy_blas(num_threads=None):
    """Smoke-check that numpy BLAS is multithreaded.

    Runs a 4096^2 x 4096 matmul and checks it completes in reasonable time. Assumes ~50
    GFLOPS/thread/s for a well-configured BLAS. Prints a warning and returns False if
    too slow.
    """
    import time

    if num_threads is None:
        num_threads = os.cpu_count() or 4
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))

    N = 4096
    flops = 2.0 * N * N * N
    # 50 GFLOPS/thread/s is conservative for modern x86 with AVX2/AVX-512
    expected_gflops = 50.0 * num_threads
    max_ms = flops / (expected_gflops * 1e9) * 1000 * 3  # 3x margin

    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    t0 = time.time()
    _ = A @ B
    elapsed_ms = (time.time() - t0) * 1000
    actual_gflops = flops / (elapsed_ms * 1e6)

    ok = elapsed_ms < max_ms
    if not ok:
        print(
            f"WARNING: numpy BLAS is slow! 4k matmul: {elapsed_ms:.0f} ms "
            f"({actual_gflops:.0f} GFLOPS), expected < {max_ms:.0f} ms "
            f"({expected_gflops:.0f} GFLOPS with {num_threads} threads).\n"
            f"  Check OPENBLAS_NUM_THREADS or install a fast BLAS (openblas, mkl)."
        )
    else:
        print(
            f"numpy BLAS ok: 4k matmul {elapsed_ms:.0f} ms "
            f"({actual_gflops:.0f} GFLOPS, {num_threads} threads)"
        )
    return ok


def detect_num_gpus():
    """Detect number of available GPUs via HIP.

    Returns 1 if detection fails (safe fallback to sequential execution).
    """
    try:
        from aster.testing import hip_get_device_count

        return max(1, hip_get_device_count())
    except Exception:
        return 1


def format_mlir_error(e):
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
    return str(e) or type(e).__name__


def parse_result_from_output(stdout):
    """Extract BENCH_RESULT_JSON line from subprocess stdout.

    Returns dict or None.
    """
    for line in stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL) :])
    return None


def compile_one(cfg, hsaco_dir, compile_fn):
    """Compile a single config to HSACO.

    Called in worker process. Returns (label, hsaco_path, KernelResources|None). Raises
    RuntimeError with actionable diagnostics on failure.
    """
    from aster.hip import parse_asm_kernel_resources, compute_register_budget

    kname = cfg.kernel_name
    output_path = os.path.join(hsaco_dir, f"{cfg.label}.hsaco")

    # Compute register budget from config's occupancy target.
    num_wg_per_cu = getattr(cfg, "num_wg_per_cu", 1) or 1
    mcpu = getattr(cfg, "mcpu", "gfx942")
    budget_vgprs, budget_agprs, _lds = compute_register_budget(
        cfg.num_threads, mcpu=mcpu, num_wg_per_cu=num_wg_per_cu
    )

    try:
        _, asm = compile_fn(
            cfg, output_path, num_vgprs=budget_vgprs, num_agprs=budget_agprs
        )
    except Exception as e:
        raise RuntimeError(format_mlir_error(e)) from None
    asm_path = output_path.replace(".hsaco", ".s")
    with open(asm_path, "w") as f:
        f.write(asm)
    res = parse_asm_kernel_resources(asm, kernel_name=kname).get(kname)
    return cfg.label, output_path, res


def exec_one_config(
    cfg, hsaco_path, num_iterations, test_dir, gpu_id, cfg_to_cli_args, script_path
):
    """Execute a single config in a subprocess pinned to a specific GPU.

    Returns (cfg, result_data, error_string). On success error_string is None; on
    failure result_data is None.
    """
    cmd = [
        sys.executable,
        script_path,
        *cfg_to_cli_args(cfg),
        "--iterations",
        str(num_iterations),
        "--hsaco",
        hsaco_path,
    ]
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

    result_data = parse_result_from_output(proc.stdout)
    if result_data is None:
        tail = ""
        if proc.stdout.strip():
            tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
        return cfg, None, f"no result line in subprocess output\n{tail}"

    return cfg, result_data, None


def print_summary_table(
    results,
    failed,
    resources_map,
    repro_cmd_fn,
    num_iterations,
):
    """Print sorted results table and save error repros to file."""
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
            res_str = str(res) if res else ""
            print(
                f"{rank:>3} {cfg.label:<60} | {ms:>8.2f} | {tflops:>8.1f} "
                f"| {pct:>6.1f}% | {lds_kb:>4.0f}KB | {res_str}"
            )

    print(f"\nSummary: {len(results)} passed, {len(failed)} failed")

    if failed:
        import tempfile as _tmp

        fd, repro_path = _tmp.mkstemp(prefix="bench_errors_", suffix=".txt", dir="/tmp")
        with os.fdopen(fd, "w") as f:
            for cfg, err in failed:
                first_line = err.split("\n")[0][:200]
                f.write(f"{cfg.label}: {first_line}\n")
                f.write(f"  {repro_cmd_fn(cfg, num_iterations)}\n\n")
        print(
            f"\n{len(failed)} compilation/occupancy/execution errors "
            f"repros saved in {repro_path}"
        )


def make_inputs(cfg):
    """Create random f16 test matrices for a given config."""
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    return A, B


def bench_perf_sweep(
    configs,
    compile_fn,
    cfg_to_cli_args,
    repro_cmd_fn,
    script_path,
    top_k_to_run=None,
    full_sweep=False,
    num_gpus=None,
    compile_workers=None,
    num_iterations=NUM_ITERATIONS,
    post_compile_filter=None,
    exec_sample=0,
):
    """Run Phase 1 (parallel compile) + Phase 2 (parallel GPU exec) sweep.

    Args:
        configs: Full list of config objects (pre-filtered for divisibility etc.)
        compile_fn: (cfg, output_path) -> (path, asm)
        cfg_to_cli_args: (cfg) -> list[str] for subprocess invocation
        repro_cmd_fn: (cfg, num_iterations) -> str for human-readable repro
        script_path: __file__ of the calling bench script (for subprocess re-entry)
        top_k_to_run: Labels to run by default (empty/None = full sweep)
        full_sweep: Ignore top_k_to_run filter
        num_gpus: GPUs for Phase 2 (None = auto-detect)
        compile_workers: Parallel compile processes (None = default)
        num_iterations: Kernel launches per config
    """
    if num_gpus is None:
        num_gpus = detect_num_gpus()
    if compile_workers is None:
        compile_workers = DEFAULT_COMPILE_WORKERS
    check_numpy_blas(num_threads=compile_workers)
    if top_k_to_run is None:
        top_k_to_run = []

    results = []
    failed = []
    active = list(configs)

    if top_k_to_run and not full_sweep:
        top_set = set(top_k_to_run)
        active = [c for c in active if c.label in top_set]

    total = len(configs)
    print(f"\nRunning {len(active)}/{total} configs")
    print(f"  iterations={num_iterations}, warmup={WARMUP_ITERATIONS}")
    print(f"  compile_workers={compile_workers}, exec_gpus={num_gpus}")
    sys.stdout.flush()

    # -- Phase 1: Parallel compilation ---------------------------------
    from tqdm import tqdm

    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    print(
        f"\n--- Phase 1: Compiling {len(active)} configs ({compile_workers} workers) ---"
    )
    print(f"  hsaco_dir: {hsaco_dir}")
    sys.stdout.flush()

    hsaco_paths = {}
    resources_map = {}
    compile_failed = {}
    with ProcessPoolExecutor(max_workers=compile_workers) as pool:
        futures = {}
        for cfg in active:
            fut = pool.submit(compile_one, cfg, hsaco_dir, compile_fn)
            futures[fut] = cfg

        total_compile = len(futures)
        pbar = tqdm(total=total_compile, desc="Compiling", unit="cfg")
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                label, path, res = fut.result()
                hsaco_paths[label] = path
                if res:
                    resources_map[label] = res
            except Exception as e:
                err = str(e)
                compile_failed[cfg.label] = err
                failed.append((cfg, f"compile: {err}"))
            pbar.update(1)
            pbar.set_postfix(ok=len(hsaco_paths), fail=len(compile_failed))
        pbar.close()

    compiled_count = len(hsaco_paths)
    print(
        f"Compilation done: {compiled_count} succeeded, "
        f"{len(compile_failed)} failed"
    )
    sys.stdout.flush()

    # -- Post-compile filter (e.g. VGPR occupancy check) ------------------
    if post_compile_filter is not None:
        before = len(hsaco_paths)
        filtered_labels = []
        for c in active:
            if c.label not in hsaco_paths:
                continue
            res = resources_map.get(c.label)
            if res and not post_compile_filter(c, res):
                filtered_labels.append(c.label)
                del hsaco_paths[c.label]
        if filtered_labels:
            print(
                f"\nPost-compile filter: {len(filtered_labels)} configs "
                f"skipped (e.g. VGPR limit)"
            )

    # -- Phase 2: Parallel execution across GPUs -------------------------
    all_compiled = [c for c in active if c.label in hsaco_paths]
    exec_active = all_compiled
    if exec_sample > 0 and len(exec_active) > exec_sample:
        import random

        exec_active = random.sample(exec_active, exec_sample)
    print(
        f"\n--- Phase 2: Executing {len(exec_active)} / {len(all_compiled)} compiled "
        f"({num_gpus} GPU(s)) ---"
    )
    sys.stdout.flush()

    test_dir = os.path.join(os.path.dirname(os.path.abspath(script_path)), "..")
    best_tflops = 0.0
    best_pct = 0.0
    exec_fail = 0

    with ThreadPoolExecutor(max_workers=num_gpus) as exec_pool:
        future_to_cfg = {}
        for i, cfg in enumerate(exec_active):
            gpu_id = i % num_gpus
            hsaco_path = hsaco_paths[cfg.label]
            fut = exec_pool.submit(
                exec_one_config,
                cfg,
                hsaco_path,
                num_iterations,
                test_dir,
                gpu_id,
                cfg_to_cli_args,
                script_path,
            )
            future_to_cfg[fut] = cfg

        pbar = tqdm(total=len(exec_active), desc="Executing", unit="cfg")
        for fut in as_completed(future_to_cfg):
            cfg = future_to_cfg[fut]
            cfg_ret, result_data, err = fut.result()
            if err is not None:
                failed.append((cfg, err))
                exec_fail += 1
            else:
                min_ms = result_data["min_ms"]
                tflops = result_data["tflops"]
                pct_peak = result_data["pct_peak"]
                results.append((cfg, min_ms, tflops, pct_peak))
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_pct = pct_peak
            pbar.update(1)
            pbar.set_postfix_str(
                f"best {best_tflops:.1f} TF ({best_pct:.1f}% peak), fail={exec_fail}"
            )
        pbar.close()

    print_summary_table(
        results,
        failed,
        resources_map,
        repro_cmd_fn,
        num_iterations,
    )

    if not results:
        print("\nNo configs succeeded.")
        return results

    print(f"\nBest {min(20, len(results))} configs:")
    for rank, (cfg, ms, tflops, pct) in enumerate(results[:20], 1):
        print(f"  #{rank} {cfg.label}: {tflops:.1f} TFLOPS ({pct:.1f}% peak)")

    return results


def print_config(cfg, iterations, resources=None):
    """Print config summary after compilation."""
    print(f"Config: {cfg.label}")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    print(f"  per-wave tiles: {cfg.m_tiles}x{cfg.n_tiles}")
    if resources:
        print(f"  resources: {resources}")
    print(f"  iterations={iterations}, warmup={WARMUP_ITERATIONS}")
    sys.stdout.flush()


def run_single(cfg, compile_fn, args, execute_fn):
    """Run a single config from CLI args.

    Emits BENCH_RESULT_JSON for sweep parsing.
    """
    from aster.hip import parse_asm_kernel_resources, compute_register_budget

    kname = cfg.kernel_name
    print_ir = getattr(args, "print_ir_after_all", False)
    print_asm = getattr(args, "print_asm", False)

    # Compute register budget from occupancy target.
    num_wg_per_cu = getattr(args, "num_wg_per_cu", 1) or 1
    mcpu = getattr(cfg, "mcpu", "gfx942")
    budget_vgprs, budget_agprs, _lds = compute_register_budget(
        cfg.num_threads, mcpu=mcpu, num_wg_per_cu=num_wg_per_cu
    )
    num_vgprs = getattr(args, "num_vgprs", None) or budget_vgprs
    num_agprs = getattr(args, "num_agprs", None) or budget_agprs

    compile_kwargs = dict(
        print_ir_after_all=print_ir, num_vgprs=num_vgprs, num_agprs=num_agprs
    )
    print(
        f"  register budget: vgpr={num_vgprs}, agpr={num_agprs}"
        f" (wg_per_cu={num_wg_per_cu})"
    )

    if args.compile_only:
        if not args.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        _, asm = compile_fn(cfg, args.hsaco, **compile_kwargs)
        resources = parse_asm_kernel_resources(asm, kernel_name=kname)
        res = resources.get(kname)
        print_config(cfg, args.iterations, res)
        if res:
            for v in res.check_occupancy(cfg.num_threads):
                print(f"  OCCUPANCY ERROR: {v}")
        print(f"  Compiled: {args.hsaco}")
        if print_asm:
            print(f"\n--- Assembly ---\n{asm}")
        return

    A, B = make_inputs(cfg)

    if args.hsaco:
        asm_path = args.hsaco.replace(".hsaco", ".s")
        asm_content = None
        res = None
        if os.path.exists(asm_path):
            with open(asm_path) as f:
                asm_content = f.read()
                res = parse_asm_kernel_resources(asm_content, kernel_name=kname).get(
                    kname
                )
        print_config(cfg, args.iterations, res)
        if print_asm and asm_content:
            print(f"\n--- Assembly ---\n{asm_content}")
        _, times_ns = execute_fn(
            cfg, args.hsaco, args.iterations, A, B, skip_gpu_check=True
        )
    else:
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            _, asm = compile_fn(cfg, tmp.name, **compile_kwargs)
            resources = parse_asm_kernel_resources(asm, kernel_name=kname)
            res = resources.get(kname)
            print_config(cfg, args.iterations, res)
            if res:
                violations = res.check_occupancy(cfg.num_threads)
                for v in violations:
                    print(f"  OCCUPANCY ERROR: {v}")
                if violations and not getattr(args, "force", False):
                    raise SystemExit(1)
            if print_asm:
                print(f"\n--- Assembly ---\n{asm}")
            _, times_ns = execute_fn(cfg, tmp.name, args.iterations, A, B)

    measured = times_ns[WARMUP_ITERATIONS:]
    min_ns = min(measured)
    min_ms = min_ns / 1e6
    tflops = cfg.total_flops / min_ns * 1e-3
    pct_peak = tflops / MI300X_PEAK_TFLOPS_F16 * 100

    print(f"\nAll iterations (ms): {[f'{t/1e6:.2f}' for t in times_ns]}")
    print(f"Measured (post-warmup): {[f'{t/1e6:.2f}' for t in measured]}")
    print(f"Min: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}% peak)")

    print(
        RESULT_SENTINEL
        + json.dumps(
            {
                "min_ms": min_ms,
                "tflops": tflops,
                "pct_peak": pct_peak,
                "times_ms": [t / 1e6 for t in times_ns],
            }
        )
    )


def add_sweep_cli_args(parser, default_compile_workers=DEFAULT_COMPILE_WORKERS):
    """Add common sweep-mode CLI args to an argparse parser."""
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
        "--compile-sample",
        type=int,
        default=4096,
        help="Random sample of configs to compile (0 = all, default: 4096)",
    )
    parser.add_argument(
        "--exec-sample",
        type=int,
        default=2048,
        help="Random sample of compiled configs to execute (0 = all, default: 2048)",
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
        default=default_compile_workers,
        help=f"Parallel compilation processes (default: {default_compile_workers})",
    )
    parser.add_argument(
        "--no-reg-filter",
        action="store_true",
        help="Disable pre-compile register estimate filter (may cause many regalloc failures)",
    )


def add_single_cli_args(parser, num_iterations=NUM_ITERATIONS):
    """Add common single-config CLI args (iterations, hsaco, compile-only)."""
    parser.add_argument(
        "--iterations",
        type=int,
        default=num_iterations,
        help=f"Kernel launches (default: {num_iterations})",
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
    parser.add_argument(
        "--print-ir-after-all",
        action="store_true",
        help="Print MLIR IR after each pass (single-config mode only)",
    )
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Print generated assembly to stdout (single-config mode only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run despite occupancy violations (use to confirm HIP will crash)",
    )
    parser.add_argument(
        "--num-vgprs",
        type=int,
        default=None,
        help="Max VGPRs for register allocation (default: computed from occupancy)",
    )
    parser.add_argument(
        "--num-agprs",
        type=int,
        default=None,
        help="Max AGPRs for register allocation (default: computed from occupancy)",
    )
    parser.add_argument(
        "--num-wg-per-cu",
        type=int,
        default=1,
        help="Target workgroups per CU for register/LDS budget (default: 1)",
    )
