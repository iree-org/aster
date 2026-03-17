"""Shared benchmark harness for weak-scaled GEMM sweeps.

Phase 1: Parallel compilation (ProcessPoolExecutor) -> HSACOs.
Phase 2: Parallel GPU execution (ProcessPoolExecutor, crash-isolated).
Phase 3: Correctness verification (ProcessPoolExecutor).
"""

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

RESULT_SENTINEL = "BENCH_RESULT_JSON:"
MI300X_PEAK_TFLOPS_F16 = 1307.0
NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2
DEFAULT_COMPILE_WORKERS = 8


# -- Helpers ---------------------------------------------------------------


def _save_tmpfile(prefix, lines):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def check_numpy_blas(label=""):
    import time

    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
    a = np.random.randn(4096, 4096).astype(np.float32)
    t0 = time.time()
    _ = a @ a
    dt = time.time() - t0
    tag = f"[{label}] " if label else ""
    if dt > 1.0:
        raise RuntimeError(
            f"{tag}numpy BLAS too slow: {dt:.1f}s. "
            f"Set OPENBLAS_NUM_THREADS={os.cpu_count()}"
        )
    print(f"{tag}numpy BLAS ok: {dt * 1000:.0f} ms")


def detect_num_gpus():
    try:
        from aster.testing import hip_get_device_count

        return max(1, hip_get_device_count())
    except Exception:
        return 1


def format_mlir_error(e):
    parts = []
    for diag in getattr(e, "error_diagnostics", []):
        msg = getattr(diag, "message", "")
        if msg:
            parts.append(msg)
        for note in getattr(diag, "notes", []):
            if getattr(note, "message", ""):
                parts.append(f"  note: {note.message}")
    return "\n".join(parts) if parts else (str(e) or type(e).__name__)


# -- Compilation (subprocess) ----------------------------------------------


def compile_one(cfg, hsaco_dir, compile_fn):
    """Compile one config to HSACO (called in worker process)."""
    from aster.hip import parse_asm_kernel_resources, compute_register_budget

    output = os.path.join(hsaco_dir, f"{cfg.label}.hsaco")
    wg = getattr(cfg, "num_wg_per_cu", 1) or 1
    bv, ba, _ = compute_register_budget(
        cfg.num_threads, mcpu=getattr(cfg, "mcpu", "gfx942"), num_wg_per_cu=wg
    )
    try:
        _, asm = compile_fn(cfg, output, num_vgprs=bv, num_agprs=ba)
    except Exception as e:
        raise RuntimeError(format_mlir_error(e)) from None
    with open(output.replace(".hsaco", ".s"), "w") as f:
        f.write(asm)
    res = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name).get(
        cfg.kernel_name
    )
    return cfg.label, output, res


# -- GPU execution (subprocess, crash-isolated) ----------------------------


def _exec_worker(args):
    """Run one HSACO in a subprocess.

    HIP_VISIBLE_DEVICES must be set by initializer.
    """
    from aster.hip import execute_hsaco

    label, hsaco_path, kernel_name, num_wg, num_threads, m, n, k, num_iter = args
    try:
        A = np.empty(m * k, dtype=np.float16)
        B = np.empty(n * k, dtype=np.float16)
        C = np.zeros(m * n, dtype=np.float32)
        times = execute_hsaco(
            hsaco_path=hsaco_path,
            kernel_name=kernel_name,
            input_arrays=[A, B],
            output_arrays=[C],
            grid_dim=(num_wg, 1, 1),
            block_dim=(num_threads, 1, 1),
            num_iterations=num_iter,
        )
        return label, times, None
    except Exception as e:
        return label, None, str(e).split("\n")[0][:200]


def _verify_worker(args):
    """Run one HSACO + compare against numpy.

    HIP_VISIBLE_DEVICES set by initializer.
    """
    from aster.hip import execute_hsaco

    label, hsaco_path, kernel_name, num_wg, num_threads, m, n, k = args
    try:
        np.random.seed(42)
        A = (np.random.randn(m, k) * 0.1).astype(np.float16)
        B = (np.random.randn(n, k) * 0.1).astype(np.float16)
        C = np.zeros(m * n, dtype=np.float32)
        execute_hsaco(
            hsaco_path=hsaco_path,
            kernel_name=kernel_name,
            input_arrays=[A.flatten(), B.flatten()],
            output_arrays=[C],
            grid_dim=(num_wg, 1, 1),
            block_dim=(num_threads, 1, 1),
            num_iterations=1,
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
        return label, None
    except AssertionError:
        diff = float(np.max(np.abs(C - expected)))
        return label, f"numeric: max_abs_diff={diff:.6g}"
    except Exception as e:
        return label, str(e).split("\n")[0][:200]


def _gpu_init(gpu_id):
    """Process pool initializer: pin this worker to a specific GPU."""
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)


def run_on_gpus(configs, hsaco_paths, num_iterations, num_gpus, desc="Running"):
    """Execute configs in subprocesses, one pool per GPU (crash-isolated).

    Each GPU gets a dedicated process pool. Workers within a pool are pinned to that GPU
    via HIP_VISIBLE_DEVICES set in the initializer.
    """
    import multiprocessing as mp
    from tqdm import tqdm

    # Round-robin configs across GPUs.
    per_gpu = [[] for _ in range(num_gpus)]
    cfg_by_label = {}
    for i, cfg in enumerate(configs):
        if cfg.label in hsaco_paths:
            per_gpu[i % num_gpus].append(
                (
                    cfg.label,
                    hsaco_paths[cfg.label],
                    cfg.kernel_name,
                    cfg.num_workgroups,
                    cfg.num_threads,
                    cfg.m_dim,
                    cfg.n_dim,
                    cfg.k,
                    num_iterations,
                )
            )
            cfg_by_label[cfg.label] = cfg

    total = sum(len(g) for g in per_gpu)
    results, failed = [], []
    best_tf = 0.0
    pbar = tqdm(total=total, desc=desc, unit="cfg")
    ctx = mp.get_context("spawn")

    for gpu_id in range(num_gpus):
        if not per_gpu[gpu_id]:
            continue
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_gpu_init,
            initargs=(gpu_id,),
        ) as pool:
            futs = {pool.submit(_exec_worker, a): a[0] for a in per_gpu[gpu_id]}
            for fut in as_completed(futs):
                label = futs[fut]
                cfg = cfg_by_label[label]
                try:
                    _, times, err = fut.result()
                except Exception as e:
                    err = str(e).split("\n")[0][:200]
                    times = None
                if err or times is None:
                    failed.append((cfg, err or "unknown"))
                else:
                    ns = min(times[WARMUP_ITERATIONS:])
                    tf = cfg.total_flops / ns * 1e-3
                    pct = tf / MI300X_PEAK_TFLOPS_F16 * 100
                    results.append((cfg, ns / 1e6, tf, pct))
                    if tf > best_tf:
                        best_tf = tf
                pbar.update(1)
                pbar.set_postfix_str(f"best {best_tf:.1f} TF, fail={len(failed)}")
    pbar.close()
    return results, failed


def verify_on_gpus(configs, hsaco_paths, num_gpus, desc="Verifying"):
    """Verify configs against numpy in subprocesses (same spawn pattern as run_on_gpus)."""
    import multiprocessing as mp
    from tqdm import tqdm

    per_gpu = [[] for _ in range(num_gpus)]
    for i, cfg in enumerate(configs):
        if cfg.label in hsaco_paths:
            per_gpu[i % num_gpus].append(
                (
                    cfg.label,
                    hsaco_paths[cfg.label],
                    cfg.kernel_name,
                    cfg.num_workgroups,
                    cfg.num_threads,
                    cfg.m_dim,
                    cfg.n_dim,
                    cfg.k,
                )
            )

    total = sum(len(g) for g in per_gpu)
    passed, errors = 0, []
    pbar = tqdm(total=total, desc=desc, unit="cfg")
    ctx = mp.get_context("spawn")

    for gpu_id in range(num_gpus):
        if not per_gpu[gpu_id]:
            continue
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_gpu_init,
            initargs=(gpu_id,),
        ) as pool:
            futs = {pool.submit(_verify_worker, a): a[0] for a in per_gpu[gpu_id]}
            for fut in as_completed(futs):
                label = futs[fut]
                try:
                    _, err = fut.result()
                except Exception as e:
                    err = str(e).split("\n")[0][:200]
                if err:
                    errors.append(f"{label}: {err}")
                else:
                    passed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"pass={passed}, fail={len(errors)}")
    pbar.close()
    return passed, errors


# -- Sweep -----------------------------------------------------------------


def bench_perf_sweep(
    configs,
    compile_fn,
    repro_cmd_fn,
    top_k_to_run=None,
    full_sweep=False,
    num_gpus=None,
    compile_workers=None,
    num_iterations=NUM_ITERATIONS,
    post_compile_filter=None,
    exec_sample=0,
):
    """Phase 1 (compile) + Phase 2 (execute).

    Returns (results, hsaco_paths).
    """
    from tqdm import tqdm

    if num_gpus is None:
        num_gpus = detect_num_gpus()
    if compile_workers is None:
        compile_workers = DEFAULT_COMPILE_WORKERS
    check_numpy_blas(label="sweep")

    active = list(configs)
    if top_k_to_run and not full_sweep:
        active = [c for c in active if c.label in set(top_k_to_run)]

    print(
        f"\nSweep: {len(active)}/{len(configs)} configs, "
        f"{compile_workers} workers, {num_gpus} GPU(s)"
    )
    sys.stdout.flush()

    # Phase 1: compile.
    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    hsaco_paths, resources_map, failed = {}, {}, []
    with ProcessPoolExecutor(max_workers=compile_workers) as pool:
        futs = {pool.submit(compile_one, c, hsaco_dir, compile_fn): c for c in active}
        pbar = tqdm(total=len(futs), desc="Compiling", unit="cfg")
        for fut in as_completed(futs):
            cfg = futs[fut]
            try:
                label, path, res = fut.result()
                hsaco_paths[label] = path
                if res:
                    resources_map[label] = res
            except Exception as e:
                failed.append((cfg, f"compile: {e}"))
            pbar.update(1)
            pbar.set_postfix(ok=len(hsaco_paths), fail=len(failed))
        pbar.close()
    print(f"Compiled: {len(hsaco_paths)} ok, {len(failed)} failed")

    # Post-compile filter.
    if post_compile_filter:
        before = len(hsaco_paths)
        for c in active:
            res = resources_map.get(c.label)
            if c.label in hsaco_paths and res and not post_compile_filter(c, res):
                del hsaco_paths[c.label]
        dropped = before - len(hsaco_paths)
        if dropped:
            print(f"Post-compile filter: {dropped} skipped")

    # Phase 2: execute in subprocesses (crash-isolated).
    exec_active = [c for c in active if c.label in hsaco_paths]
    if exec_sample > 0 and len(exec_active) > exec_sample:
        import random

        exec_active = random.sample(exec_active, exec_sample)

    print(f"\n--- Executing {len(exec_active)} configs ({num_gpus} GPU(s)) ---")
    results, exec_failed = run_on_gpus(
        exec_active,
        hsaco_paths,
        num_iterations,
        num_gpus,
        desc="Executing",
    )
    failed.extend(exec_failed)

    # Summary: separate files for compile errors vs exec errors.
    results.sort(key=lambda r: r[2], reverse=True)
    compile_errs = [(c, e) for c, e in failed if e.startswith("compile:")]
    exec_errs = [(c, e) for c, e in failed if not e.startswith("compile:")]

    if results:
        lines = [
            f"#{i+1:>3} {tf:>7.1f} TF {pct:>5.1f}% {ms:>8.2f}ms {c.label}"
            for i, (c, ms, tf, pct) in enumerate(results)
        ]
        print(
            f"\nResults ({len(results)}) saved in {_save_tmpfile('bench_results_', lines)}"
        )
    if compile_errs:
        lines = [f"{c.label}: {e}" for c, e in compile_errs]
        print(
            f"{len(compile_errs)} compile errors in {_save_tmpfile('bench_compile_errors_', lines)}"
        )
    if exec_errs:
        lines = [f"{c.label}: {e}" for c, e in exec_errs]
        print(
            f"{len(exec_errs)} exec errors in {_save_tmpfile('bench_exec_errors_', lines)}"
        )

    print(
        f"\nSummary: {len(results)} ok, {len(compile_errs)} compile fail, {len(exec_errs)} exec fail"
    )
    if results:
        top_n = min(20, len(results))
        print(f"Top {top_n}:")
        for i, (c, ms, tf, pct) in enumerate(results[:top_n]):
            print(f"  #{i+1} {c.label}: {tf:.1f} TF ({pct:.1f}%)")

    return results, hsaco_paths


# -- Single-config mode ----------------------------------------------------


def make_inputs(cfg):
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    return A, B


def print_config(cfg, iterations, resources=None):
    print(f"Config: {cfg.label}")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    if resources:
        print(f"  resources: {resources}")


def run_single(cfg, compile_fn, args, execute_fn):
    from aster.hip import parse_asm_kernel_resources, compute_register_budget

    kname = cfg.kernel_name
    print_ir = getattr(args, "print_ir_after_all", False)
    print_asm = getattr(args, "print_asm", False)

    wg = getattr(args, "num_wg_per_cu", 1) or 1
    bv, ba, _ = compute_register_budget(
        cfg.num_threads, mcpu=getattr(cfg, "mcpu", "gfx942"), num_wg_per_cu=wg
    )
    nv = getattr(args, "num_vgprs", None) or bv
    na = getattr(args, "num_agprs", None) or ba
    compile_kw = dict(print_ir_after_all=print_ir, num_vgprs=nv, num_agprs=na)
    print(f"  register budget: vgpr={nv}, agpr={na} (wg_per_cu={wg})")

    if args.compile_only:
        if not args.hsaco:
            raise SystemExit("--compile-only requires --hsaco")
        _, asm = compile_fn(cfg, args.hsaco, **compile_kw)
        res = parse_asm_kernel_resources(asm, kernel_name=kname).get(kname)
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
        print_config(cfg, args.iterations)
        _, times_ns = execute_fn(
            cfg, args.hsaco, args.iterations, A, B, skip_gpu_check=True
        )
    else:
        import tempfile as _tmp

        with _tmp.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            _, asm = compile_fn(cfg, tmp.name, **compile_kw)
            res = parse_asm_kernel_resources(asm, kernel_name=kname).get(kname)
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
    tf = cfg.total_flops / min_ns * 1e-3
    pct = tf / MI300X_PEAK_TFLOPS_F16 * 100
    print(f"\nMin: {min_ns/1e6:.2f} ms  {tf:.1f} TFLOPS  ({pct:.1f}% peak)")
    print(
        RESULT_SENTINEL
        + json.dumps(
            {
                "min_ms": min_ns / 1e6,
                "tflops": tf,
                "pct_peak": pct,
                "times_ms": [t / 1e6 for t in times_ns],
            }
        )
    )


# -- CLI args --------------------------------------------------------------


def add_sweep_cli_args(parser):
    a = parser.add_argument
    a("--sweep", action="store_true", help="Run sweep")
    a("--full-sweep", action="store_true", help="All configs (no top-k filter)")
    a("--compile-sample", type=int, default=4096, help="Configs to compile (0=all)")
    a("--exec-sample", type=int, default=2048, help="Configs to execute (0=all)")
    a("--num-gpus", type=int, default=None, help="GPUs (default: auto)")
    a("--compile-workers", type=int, default=DEFAULT_COMPILE_WORKERS)
    a("--no-reg-filter", action="store_true", help="Disable register estimate filter")


def add_single_cli_args(parser, num_iterations=NUM_ITERATIONS):
    a = parser.add_argument
    a("--iterations", type=int, default=num_iterations)
    a("--hsaco", type=str, default=None, help="Pre-compiled HSACO path")
    a("--compile-only", action="store_true")
    a("--print-ir-after-all", action="store_true")
    a("--print-asm", action="store_true")
    a("--force", action="store_true", help="Run despite occupancy violations")
    a("--num-vgprs", type=int, default=None)
    a("--num-agprs", type=int, default=None)
    a("--num-wg-per-cu", type=int, default=1)
