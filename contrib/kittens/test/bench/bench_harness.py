"""Shared benchmark harness for weak-scaled GEMM sweeps.

Phase 1: Parallel compilation (ProcessPoolExecutor) -> HSACOs.
Phase 2: Parallel GPU execution (ProcessPoolExecutor, crash-isolated).
Phase 3: Correctness verification (ProcessPoolExecutor).
"""

import json
import os
import signal
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
DEFAULT_COMPILE_TIMEOUT = 60  # seconds per kernel


# -- Helpers ---------------------------------------------------------------


def _save_tmpfile(prefix, lines):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _save_error_file(prefix, phase, errors, repro_cmd_fn=None, num_iterations=None):
    """Save errors grouped by category for easy debugging.

    Output format:
      - Top-level summary (total count, unique categories)
      - One section per category, most frequent first
      - Each section has a header and all configs in that category
    """
    from collections import defaultdict

    by_category = defaultdict(list)
    for c, e, full in errors:
        by_category[e].append((c, full))

    lines = [
        f"# {len(errors)} {phase} failures, {len(by_category)} unique errors",
        "#",
    ]
    for msg, entries in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"# {len(entries):>5}x {msg}")
    lines.append("")

    for msg, entries in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        lines.append("=" * 78)
        lines.append(f"[{len(entries)}x] {msg}")
        lines.append("=" * 78)
        lines.append("")
        for c, full in entries:
            repro = ""
            if repro_cmd_fn:
                try:
                    repro = f" | repro: {repro_cmd_fn(c, num_iterations)}"
                except Exception:
                    pass
            lines.append(f"  {c.label}{repro}")
            if full and full != msg.removeprefix(f"{phase}: "):
                for fline in full.split("\n"):
                    lines.append(f"    {fline}")
        lines.append("")

    return _save_tmpfile(prefix, lines)


def make_sweep_filter(args, attr_map):
    """Build a config filter from CLI args that pin sweep dimensions.

    Args:
        args: Parsed argparse namespace.
        attr_map: Dict mapping argparse attribute names to config attribute names.
            E.g. {"stages": "num_stages", "m_waves": "m_waves"}.
            If a CLI arg is None, it is ignored (not pinned).

    Returns:
        A predicate (cfg) -> bool that returns True if cfg matches all pinned values,
        or None if no dimensions are pinned.
    """
    pins = {}
    for arg_name, cfg_attr in attr_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            pins[cfg_attr] = val
    if not pins:
        return None
    desc = ", ".join(f"{k}={v}" for k, v in pins.items())
    print(f"Sweep filter: {desc}")
    return lambda cfg: all(getattr(cfg, k) == v for k, v in pins.items())


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
    """Return the number of available GPUs, or 0 if none are present."""
    try:
        from aster.hip import system_has_gpu

        if not system_has_gpu("gfx942"):
            return 0
        from aster.testing import hip_get_device_count

        return max(1, hip_get_device_count())
    except Exception:
        return 0


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


def format_mlir_error_oneline(e):
    """Extract a single-line error summary from an MLIR exception."""
    full = format_mlir_error(e)
    first = full.split("\n")[0].strip()
    return first[:200] if first else type(e).__name__


# -- Compilation (subprocess, crash-isolated) ------------------------------


def _compile_inner(cfg, hsaco_dir, compile_fn, result_pipe, stderr_path):
    """Run compilation in an isolated child process. Sends result via pipe.

    If this process crashes (segfault, assertion), the parent reads stderr_path to
    capture the error spew. stderr is redirected to a file so it survives crashes (pipes
    would lose buffered data on SIGKILL/SIGSEGV).
    """
    # Redirect stderr to file so crash output is preserved.
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(stderr_fd, 2)
    os.close(stderr_fd)

    try:
        from aster.hip import parse_asm_kernel_resources, compute_register_budget

        output = os.path.join(hsaco_dir, f"{cfg.label}.hsaco")
        wg = getattr(cfg, "num_wg_per_cu", 1) or 1
        bv, ba, _ = compute_register_budget(
            cfg.num_threads, mcpu=getattr(cfg, "mcpu", "gfx942"), num_wg_per_cu=wg
        )
        _, asm = compile_fn(cfg, output, num_vgprs=bv, num_agprs=ba)
        with open(output.replace(".hsaco", ".s"), "w") as f:
            f.write(asm)
        res = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name).get(
            cfg.kernel_name
        )
        result_pipe.send(("ok", (cfg.label, output, res)))
    except Exception as e:
        result_pipe.send(("error", format_mlir_error_oneline(e)))
    finally:
        result_pipe.close()


def _read_stderr_log(path, max_bytes=4096):
    """Read the tail of a stderr log file, return as string."""
    try:
        size = os.path.getsize(path)
        with open(path) as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # skip partial line
            return f.read().strip()
    except Exception:
        return ""


def compile_one(cfg, hsaco_dir, compile_fn, timeout=DEFAULT_COMPILE_TIMEOUT):
    """Compile one config to HSACO in a crash-isolated subprocess.

    Spawns a child process for the actual compilation. If it crashes (segfault,
    assertion) or exceeds the timeout, the pool worker stays alive and reports the
    failure. Crash stderr is captured to a log file in hsaco_dir.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    stderr_path = os.path.join(hsaco_dir, f"{cfg.label}.stderr")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_compile_inner,
        args=(cfg, hsaco_dir, compile_fn, child_conn, stderr_path),
    )
    p.start()
    child_conn.close()

    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        parent_conn.close()
        stderr = _read_stderr_log(stderr_path)
        msg = f"compilation timed out after {timeout}s"
        if stderr:
            msg += f"\n{stderr}"
        raise TimeoutError(msg)

    if p.exitcode != 0:
        parent_conn.close()
        stderr = _read_stderr_log(stderr_path)
        sig = -p.exitcode if p.exitcode < 0 else p.exitcode
        msg = f"crash (signal {sig})" if p.exitcode < 0 else f"crash (exit {sig})"
        if stderr:
            msg += f"\n{stderr}"
        raise RuntimeError(msg)

    # Clean up stderr log on success.
    try:
        os.unlink(stderr_path)
    except OSError:
        pass

    if not parent_conn.poll():
        parent_conn.close()
        raise RuntimeError("compilation produced no result")

    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "error":
        raise RuntimeError(payload)
    return payload


# -- GPU execution (subprocess, crash-isolated) ----------------------------


def _exec_worker(args):
    """Run one HSACO in a subprocess.

    HIP_VISIBLE_DEVICES and stderr suppression set by _gpu_init initializer.
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

    HIP_VISIBLE_DEVICES and stderr suppression set by _gpu_init initializer.
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
    """Process pool initializer: pin worker to a GPU and silence all native output.

    HIP/HSA runtime prints crash, queue-reset, and debug messages through multiple
    channels (fd 1, fd 2, AMD logging, HSA tools). We suppress all of them here so
    nothing leaks to the parent terminal. Error info comes back via Python exceptions.
    """
    import io

    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    # Suppress AMD/HIP/HSA logging at the source.
    os.environ["AMD_LOG_LEVEL"] = "0"
    os.environ["HIP_TRACE_API"] = "0"
    os.environ["HSA_TOOLS_LIB"] = ""
    # Redirect both C-level stdout (fd 1) and stderr (fd 2) to /dev/null.
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    # Also redirect Python-level streams (some libraries use sys.stderr directly).
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()


def run_on_gpus(configs, hsaco_paths, num_iterations, num_gpus, desc="Running"):
    """Execute configs in subprocesses, all GPUs concurrently (crash-isolated).

    Each GPU gets a dedicated process pool with max_workers=1. All pools run
    concurrently and results are collected as they complete across all GPUs.
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

    # Launch all GPU pools concurrently.
    pools = []
    all_futs = {}
    try:
        for gpu_id in range(num_gpus):
            if not per_gpu[gpu_id]:
                continue
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_gpu_init,
                initargs=(gpu_id,),
            )
            pools.append(pool)
            for a in per_gpu[gpu_id]:
                all_futs[pool.submit(_exec_worker, a)] = a[0]

        for fut in as_completed(all_futs):
            label = all_futs[fut]
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
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping execution, moving to reporting...")
        for f in all_futs:
            f.cancel()
    finally:
        for pool in pools:
            pool.shutdown(wait=False, cancel_futures=True)
    pbar.close()
    return results, failed


def verify_on_gpus(configs, hsaco_paths, num_gpus, desc="Verifying"):
    """Verify configs against numpy in subprocesses, all GPUs concurrently.

    Each worker computes its own reference matmul (no large-array IPC needed). With
    concurrent GPU pools, CPU reference work on one GPU's worker overlaps with GPU
    execution on other GPUs.
    """
    import multiprocessing as mp
    from tqdm import tqdm

    # Build lightweight work items (no precomputed arrays -- workers handle it).
    per_gpu = [[] for _ in range(num_gpus)]
    idx = 0
    for cfg in configs:
        if cfg.label not in hsaco_paths:
            continue
        item = (
            cfg.label,
            hsaco_paths[cfg.label],
            cfg.kernel_name,
            cfg.num_workgroups,
            cfg.num_threads,
            cfg.m_dim,
            cfg.n_dim,
            cfg.k,
        )
        per_gpu[idx % num_gpus].append(item)
        idx += 1

    total = sum(len(g) for g in per_gpu)
    passed, errors = 0, []
    pbar = tqdm(total=total, desc=desc, unit="cfg")
    ctx = mp.get_context("spawn")

    # Launch all GPU pools concurrently.
    pools = []
    all_futs = {}
    try:
        for gpu_id in range(num_gpus):
            if not per_gpu[gpu_id]:
                continue
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_gpu_init,
                initargs=(gpu_id,),
            )
            pools.append(pool)
            for a in per_gpu[gpu_id]:
                all_futs[pool.submit(_verify_worker, a)] = a[0]

        for fut in as_completed(all_futs):
            label = all_futs[fut]
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
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping verification, reporting partial results...")
        for f in all_futs:
            f.cancel()
    finally:
        for pool in pools:
            pool.shutdown(wait=False, cancel_futures=True)
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
    compile_timeout=DEFAULT_COMPILE_TIMEOUT,
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
    import multiprocessing as mp

    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    hsaco_paths, resources_map, failed = {}, {}, []
    spawn_ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=compile_workers, mp_context=spawn_ctx)
    futs = {
        pool.submit(compile_one, c, hsaco_dir, compile_fn, compile_timeout): c
        for c in active
    }
    pbar = tqdm(total=len(futs), desc="Compiling", unit="cfg")
    try:
        for fut in as_completed(futs):
            cfg = futs[fut]
            try:
                label, path, res = fut.result()
                hsaco_paths[label] = path
                if res:
                    resources_map[label] = res
            except Exception as e:
                full_err = str(e).strip()
                short = full_err.split("\n")[0].strip()[:200]
                if not short:
                    short = type(e).__name__
                failed.append((cfg, f"compile: {short}", full_err))
            pbar.update(1)
            pbar.set_postfix(ok=len(hsaco_paths), fail=len(failed))
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping compilation, moving to execution phase...")
        for f in futs:
            f.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
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

    if num_gpus == 0:
        print("\nNo GPUs detected -- skipping execution phase.")
        results, exec_failed = [], []
    else:
        print(f"\n--- Executing {len(exec_active)} configs ({num_gpus} GPU(s)) ---")
        results, exec_failed = run_on_gpus(
            exec_active,
            hsaco_paths,
            num_iterations,
            num_gpus,
            desc="Executing",
        )
    failed.extend((c, e, "") for c, e in exec_failed)

    # Summary: separate files for compile errors vs exec errors.
    results.sort(key=lambda r: r[2], reverse=True)
    compile_errs = [(c, e, full) for c, e, full in failed if e.startswith("compile:")]
    exec_errs = [(c, e, full) for c, e, full in failed if not e.startswith("compile:")]
    saved_files = []

    if results:
        lines = []
        for i, (c, ms, tf, pct) in enumerate(results):
            line = f"#{i+1:>3} {tf:>7.1f} TF {pct:>5.1f}% {ms:>8.2f}ms {c.label}"
            if repro_cmd_fn:
                try:
                    line += f" | repro: {repro_cmd_fn(c, num_iterations)}"
                except Exception:
                    pass
            lines.append(line)
        p = _save_tmpfile("bench_results_", lines)
        saved_files.append(p)
        print(f"\nResults ({len(results)}) saved in {p}")
    if compile_errs:
        p = _save_error_file(
            "bench_compile_errors_",
            "compile",
            compile_errs,
            repro_cmd_fn,
            num_iterations,
        )
        saved_files.append(p)
        print(f"{len(compile_errs)} compile errors in {p}")
    if exec_errs:
        p = _save_error_file("bench_exec_errors_", "exec", exec_errs)
        saved_files.append(p)
        print(f"{len(exec_errs)} exec errors in {p}")

    print(
        f"\nSummary: {len(results)} ok, {len(compile_errs)} compile fail, {len(exec_errs)} exec fail"
    )
    if results:
        top_n = min(20, len(results))
        print(f"Top {top_n}:")
        for i, (c, ms, tf, pct) in enumerate(results[:top_n]):
            print(f"  #{i+1} {c.label}: {tf:.1f} TF ({pct:.1f}%)")
    if saved_files:
        print(f"\nSaved files:")
        for f in saved_files:
            print(f"  {f}")

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
    compile_kw = dict(
        print_ir_after_all=print_ir, print_asm=print_asm, num_vgprs=nv, num_agprs=na
    )
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
        return

    has_gpu = detect_num_gpus() > 0

    if args.hsaco:
        if not has_gpu:
            print("No GPUs detected -- skipping execution.")
            return
        A, B = make_inputs(cfg)
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
            if not has_gpu:
                print("No GPUs detected -- skipping execution.")
                return
            A, B = make_inputs(cfg)
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
    a(
        "--compile-timeout",
        type=int,
        default=DEFAULT_COMPILE_TIMEOUT,
        help=f"Per-kernel compile timeout in seconds (default: {DEFAULT_COMPILE_TIMEOUT})",
    )
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
