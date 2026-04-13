"""Shared benchmark harness for weak-scaled GEMM sweeps.

Phase 1: Parallel compilation (ProcessPoolExecutor) -> HSACOs.
Phase 2: Parallel GPU execution (per-config subprocess, crash-isolated).
Phase 3: Correctness verification (per-config subprocess, crash-isolated).
"""

import fcntl
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from kittens.gemm_config import DIM_M, DIM_N, DIM_K

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

RESULT_SENTINEL = "BENCH_RESULT_JSON:"
MI300X_PEAK_TFLOPS_F16 = 1307.0
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 20
DEFAULT_COMPILE_WORKERS = 8
DEFAULT_COMPILE_TIMEOUT = 180  # seconds per kernel
DEFAULT_EXEC_TIMEOUT = 10  # seconds per kernel execution


# -- Helpers ---------------------------------------------------------------


def _save_tmpfile(prefix, lines):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _save_error_file(prefix, phase, errors, repro_cmd_fn=None):
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
                    repro = f" | repro: {repro_cmd_fn(c)}"
                except Exception:
                    pass
            lines.append(f"  {c.label}{repro}")
            if full and full != msg.removeprefix(f"{phase}: "):
                for fline in full.split("\n"):
                    lines.append(f"    {fline}")
        lines.append("")

    return _save_tmpfile(prefix, lines)


def make_sweep_pins(args, attr_map):
    """Build a pin dict from CLI args for vectorized sweep filtering.

    Args:
        args: Parsed argparse namespace.
        attr_map: Dict mapping argparse attribute names to config attribute names.
            E.g. {"stages": "num_stages", "waves_per_wg_m": "waves_per_wg_m"}.
            If a CLI arg is None, it is ignored (not pinned).

    Returns:
        A dict {config_attr: value} of pinned dimensions, or None if nothing pinned.
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
    return pins


def make_sweep_filter(args, attr_map):
    """Build a config filter predicate from CLI args that pin sweep dimensions.

    Prefer make_sweep_pins for vectorized filtering. This returns a
    Python callable for cases that need per-config predicate filtering.
    """
    pins = make_sweep_pins(args, attr_map)
    if pins is None:
        return None
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
        raise RuntimeError(f"{tag}numpy BLAS too slow: {dt:.1f}s. Set OPENBLAS_NUM_THREADS={os.cpu_count()}")
    print(f"{tag}numpy BLAS ok: {dt * 1000:.0f} ms")


def detect_num_gpus(mcpu: str):
    """Return the number of GPUs matching ``mcpu``, or 0 if none are present."""
    try:
        from aster.execution.utils import system_has_gpu

        if not system_has_gpu(mcpu):
            return 0
        from aster._mlir_libs._runtime_module import hip_get_device_count

        return max(1, hip_get_device_count())
    except Exception:
        return 0


def warn_mcpu_mismatch(compile_mcpu: str) -> None:
    """Warn if the installed GPU arch differs from ``compile_mcpu``."""
    try:
        from aster.core.device import try_query_device

        dev = try_query_device(0)
    except ImportError:
        dev = None
    if dev is None:
        return
    host_mcpu = dev.gcn_arch_name.split(":", 1)[0]
    if host_mcpu != compile_mcpu:
        print(
            f"WARNING: compiling for {compile_mcpu} but this host is "
            f"{host_mcpu}. Execution on this host will be skipped. Pass "
            f"--mcpu {host_mcpu} to target the host, or run remotely on a "
            f"{compile_mcpu} host.",
            file=sys.stderr,
        )


def require_gpu_or_compile_only(args) -> None:
    """Fail hard if no matching GPU is present and --compile-only was not set.

    Bench scripts call this right after parse_args() so the user sees
    the "no GPU" error immediately, before starting a long compile sweep
    that would silently produce zero measurements. With --compile-only
    the check is a no-op; execution is skipped intentionally.
    """
    if getattr(args, "compile_only", False):
        print(f"Compile-only mode ({args.mcpu}); execution will be skipped.")
        return
    n = detect_num_gpus(args.mcpu)
    if n == 0:
        sys.exit(
            f"ERROR: no {args.mcpu} GPU detected. Compilation would succeed "
            f"but execution would be silently skipped. Pass --compile-only "
            f"to compile without running, or run on a host with a matching "
            f"GPU (set --mcpu to the host's arch to cross-target)."
        )
    print(f"Detected {n} {args.mcpu} GPU(s).")


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

    If this process crashes (segfault, assertion), the parent reads
    stderr_path to capture the error spew. stderr is redirected to a
    file so it survives crashes (pipes would lose buffered data on
    SIGKILL/SIGSEGV).
    """
    # Redirect stderr to file so crash output is preserved.
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(stderr_fd, 2)
    os.close(stderr_fd)

    try:
        from aster.compiler.metadata import (
            parse_asm_kernel_resources,
            compute_register_budget,
        )

        output = os.path.join(hsaco_dir, f"{cfg.label}.hsaco")
        wg = getattr(cfg, "num_wg_per_cu", 1) or 1
        agpr_est = getattr(cfg, "estimated_agprs", 0) or 0
        bv, ba, _ = compute_register_budget(
            cfg.num_threads,
            mcpu=cfg.mapping.mcpu,
            num_wg_per_cu=wg,
            agpr_hint=agpr_est,
        )
        _, asm = compile_fn(cfg, output, num_vgprs=bv, num_agprs=ba)
        with open(output.replace(".hsaco", ".s"), "w") as f:
            f.write(asm)
        res = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name).get(cfg.kernel_name)
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

    Spawns a child process for the actual compilation. If it crashes
    (segfault, assertion) or exceeds the timeout, the pool worker stays
    alive and reports the failure. Crash stderr is captured to a log
    file in hsaco_dir.
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

    HIP_VISIBLE_DEVICES and stderr suppression set by _gpu_init
    initializer.
    """
    from aster.execution.core import execute_hsaco, InputArray, OutputArray

    label, hsaco_path, kernel_name, num_wg, num_threads, m, n, k, num_iter, *extra = args
    direct_b = extra[0] if extra else False
    direct_a = extra[1] if len(extra) > 1 else False
    use_zero_init = extra[2] if len(extra) > 2 else False
    try:
        if use_zero_init:
            A = np.zeros((m, k), dtype=np.float16)
            B = np.zeros((n, k), dtype=np.float16)
        else:
            np.random.seed(42)
            A = (np.random.randn(m, k) * 0.1).astype(np.float16)
            B = (np.random.randn(n, k) * 0.1).astype(np.float16)
        if direct_a:
            from kittens_helpers import shuffle_weight

            A = shuffle_weight(A)
        if direct_b:
            from kittens_helpers import shuffle_weight

            B = shuffle_weight(B)
        C = np.zeros(m * n, dtype=np.float32)
        times = execute_hsaco(
            hsaco_path=hsaco_path,
            kernel_name=kernel_name,
            arguments=[
                InputArray(A.flatten()),
                InputArray(B.flatten()),
                OutputArray(C),
            ],
            grid_dim=(num_wg, 1, 1),
            block_dim=(num_threads, 1, 1),
            num_iterations=num_iter,
        )
        return label, times, None
    except Exception as e:
        return label, None, str(e).split("\n")[0][:200]


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


def _exec_child(conn, item, gid, gpu_lock_path=None):
    """Child process entry point for isolated execution (module-level for pickling).

    If gpu_lock_path is set, wraps execution in an fcntl file lock that
    is auto-released by the OS on crash (prevents deadlocks in pipelined
    mode).
    """
    _gpu_init(gid)
    if gpu_lock_path:
        lock_fd = os.open(gpu_lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        result = _exec_worker(item)
    except Exception as e:
        if gpu_lock_path:
            try:
                from aster._mlir_libs._runtime_module import hip_device_reset

                hip_device_reset()
            except Exception:
                pass
        result = (item[0], None, str(e).split("\n")[0][:200])
    finally:
        if gpu_lock_path:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass
    conn.send(result)
    conn.close()


def _exec_one_isolated(work_item, gpu_id, timeout=120, gpu_lock_path=None):
    """Execute one config in a fully isolated subprocess.

    Unlike ProcessPoolExecutor, a crash here cannot poison other
    configs. If gpu_lock_path is set, the child acquires an fcntl file
    lock before GPU work (auto- released on crash). Returns (label,
    times, error_string).
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_exec_child,
        args=(child_conn, work_item, gpu_id, gpu_lock_path),
    )
    p.start()
    child_conn.close()

    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        parent_conn.close()
        return work_item[0], None, f"execution timed out after {timeout}s"

    if p.exitcode != 0:
        parent_conn.close()
        sig = -p.exitcode if p.exitcode < 0 else p.exitcode
        kind = "signal" if p.exitcode < 0 else "exit"
        return work_item[0], None, f"worker crash ({kind} {sig})"

    if not parent_conn.poll():
        parent_conn.close()
        return work_item[0], None, "execution produced no result"

    result = parent_conn.recv()
    parent_conn.close()
    return result


# ---------------------------------------------------------------------------
# Persistent GPU worker: one long-lived process per GPU, memory reuse
# ---------------------------------------------------------------------------


def _persistent_worker_loop(cmd_conn, result_conn, gpu_id):
    """Long-lived child process: init HIP once, process configs, reuse memory.

    Protocol: parent sends work_item tuples through cmd_conn.
    None sentinel = shut down cleanly.
    Results sent back through result_conn as (label, times, error_string).
    """
    _gpu_init(gpu_id)

    from aster.execution.core import (
        execute_hsaco,
        InputArray,
        OutputArray,
        MemoryManager,
    )

    mm = MemoryManager()
    # Track current GPU allocation sizes to avoid unnecessary realloc.
    cur_a_elems = cur_b_elems = cur_c_elems = 0
    host_A = host_B = host_C = None

    while True:
        try:
            item = cmd_conn.recv()
        except EOFError:
            break
        if item is None:
            break

        label, hsaco_path, kernel_name, num_wg, num_threads, m, n, k, num_iter, *extra = item
        direct_b = extra[0] if extra else False
        direct_a = extra[1] if len(extra) > 1 else False
        use_zero_init = extra[2] if len(extra) > 2 else False

        try:
            a_elems = m * k
            b_elems = n * k
            c_elems = m * n

            # Reallocate host arrays only when sizes change.
            if a_elems != cur_a_elems or b_elems != cur_b_elems or c_elems != cur_c_elems:
                mm.release_all()
                if use_zero_init:
                    host_A = np.zeros(a_elems, dtype=np.float16)
                    host_B = np.zeros(b_elems, dtype=np.float16)
                else:
                    np.random.seed(42)
                    host_A = (np.random.randn(a_elems) * 0.1).astype(np.float16)
                    host_B = (np.random.randn(b_elems) * 0.1).astype(np.float16)
                if direct_a:
                    from kittens_helpers import shuffle_weight

                    host_A = shuffle_weight(host_A.reshape(m, k)).flatten()
                if direct_b:
                    from kittens_helpers import shuffle_weight

                    host_B = shuffle_weight(host_B.reshape(n, k)).flatten()
                host_C = np.zeros(c_elems, dtype=np.float32)
                cur_a_elems, cur_b_elems, cur_c_elems = a_elems, b_elems, c_elems
            else:
                # Same size: just zero C, keep A/B.
                host_C[:] = 0

            times = execute_hsaco(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                arguments=[
                    InputArray(host_A),
                    InputArray(host_B),
                    OutputArray(host_C),
                ],
                grid_dim=(num_wg, 1, 1),
                block_dim=(num_threads, 1, 1),
                num_iterations=num_iter,
            )
            result_conn.send((label, times, None))
        except Exception as e:
            result_conn.send((label, None, str(e).split("\n")[0][:200]))

    mm.release_all()
    result_conn.close()


def _persistent_verify_loop(cmd_conn, result_conn, gpu_id):
    """Long-lived child for correctness verification: init HIP once, verify configs.

    Same protocol as _persistent_worker_loop: 3-tuples (label, None, error_or_none).
    """
    _gpu_init(gpu_id)
    from aster.execution.core import execute_hsaco, InputArray, OutputArray

    while True:
        try:
            item = cmd_conn.recv()
        except EOFError:
            break
        if item is None:
            break

        label, hsaco_path, kernel_name, num_wg, num_threads, m, n, k, *extra = item
        direct_b = extra[0] if extra else False
        direct_a = extra[1] if len(extra) > 1 else False

        try:
            np.random.seed(42)
            A = (np.random.randn(m, k) * 0.1).astype(np.float16)
            B = (np.random.randn(n, k) * 0.1).astype(np.float16)
            expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
            if direct_a:
                from kittens_helpers import shuffle_weight

                A = shuffle_weight(A)
            if direct_b:
                from kittens_helpers import shuffle_weight

                B = shuffle_weight(B)
            C = np.zeros(m * n, dtype=np.float32)
            execute_hsaco(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                arguments=[
                    InputArray(A.flatten()),
                    InputArray(B.flatten()),
                    OutputArray(C),
                ],
                grid_dim=(num_wg, 1, 1),
                block_dim=(num_threads, 1, 1),
                num_iterations=1,
            )
            np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
            result_conn.send((label, None, None))
        except AssertionError:
            diff = float(np.max(np.abs(C - expected)))
            result_conn.send((label, None, f"numeric: max_abs_diff={diff:.6g}"))
        except Exception as e:
            result_conn.send((label, None, str(e).split("\n")[0][:200]))

    result_conn.close()


class _PersistentGpuWorker:
    """Manages a long-lived child process on one GPU.

    Automatically respawns on crash.  The parent calls send(item) and
    recv() in pairs.  The child reuses HIP context and GPU memory across
    configs.
    """

    def __init__(self, gpu_id, loop_fn=None):
        self.gpu_id = gpu_id
        self._loop_fn = loop_fn or _persistent_worker_loop
        self._proc = None
        self._cmd = None  # parent -> child
        self._res = None  # child -> parent

    def _spawn(self):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        # Pipe(duplex=False) -> (reader, writer).
        # cmd channel: parent writes, child reads.
        cmd_reader, cmd_writer = ctx.Pipe(duplex=False)
        # res channel: child writes, parent reads.
        res_reader, res_writer = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=self._loop_fn,
            args=(cmd_reader, res_writer, self.gpu_id),
            daemon=True,
        )
        p.start()
        # Close the ends we don't use in the parent.
        cmd_reader.close()
        res_writer.close()
        self._proc = p
        self._cmd = cmd_writer  # parent sends commands
        self._res = res_reader  # parent reads results

    def ensure_alive(self):
        if self._proc is None or not self._proc.is_alive():
            if self._proc is not None:
                self._cleanup()
            self._spawn()

    def send(self, item):
        self.ensure_alive()
        self._cmd.send(item)

    def recv(self, timeout=120):
        if not self._res.poll(timeout):
            # Timed out -- kill and report.
            label = "unknown"
            self._proc.kill()
            self._proc.join()
            self._cleanup()
            return label, None, f"execution timed out after {timeout}s"
        return self._res.recv()

    def execute(self, item, timeout=120):
        """Send item and wait for result.

        Respawn on crash.
        """
        self.ensure_alive()
        self._cmd.send(item)
        if not self._res.poll(timeout):
            label = item[0]
            self._proc.kill()
            self._proc.join()
            self._cleanup()
            return label, None, f"execution timed out after {timeout}s"
        try:
            return self._res.recv()
        except (EOFError, ConnectionResetError):
            label = item[0]
            self._proc.join()
            exitcode = self._proc.exitcode
            self._cleanup()
            sig = -exitcode if exitcode and exitcode < 0 else exitcode
            kind = "signal" if exitcode and exitcode < 0 else "exit"
            return label, None, f"worker crash ({kind} {sig})"

    def shutdown(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._cmd.send(None)
                self._proc.join(timeout=5)
            except (BrokenPipeError, OSError):
                pass
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
        self._cleanup()

    def _cleanup(self):
        for c in (self._cmd, self._res):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass
        self._proc = None
        self._cmd = None
        self._res = None


def _run_gpu_queue(gpu_id, items, result_queue, timeout=120, gpu_lock_path=None, loop_fn=None):
    """Process work items on one GPU using a persistent worker.

    items can be a list (batch mode) or a queue.Queue (pipelined mode,
    stop on None). A single long-lived child process per GPU reuses HIP
    context and GPU memory across configs. On crash the worker is
    respawned automatically; the crashed config is reported as failed
    and the next config proceeds. Results are pushed to result_queue
    (thread-safe) as they complete.
    """
    import queue as _queue_mod

    worker = _PersistentGpuWorker(gpu_id, loop_fn=loop_fn)
    try:
        if isinstance(items, _queue_mod.Queue):
            while True:
                item = items.get()
                if item is None:
                    break
                result_queue.put(worker.execute(item, timeout=timeout))
        else:
            for item in items:
                result_queue.put(worker.execute(item, timeout=timeout))
    finally:
        worker.shutdown()


def run_on_gpus(configs, hsaco_paths, num_iterations, num_gpus, desc="Running"):
    """Execute configs in subprocesses, all GPUs concurrently (crash-isolated).

    Each GPU gets a dedicated thread that processes its queue sequentially.
    Each config within a queue runs in its own subprocess (via
    _exec_one_isolated) so a crash (GPU hang, segfault) cannot poison other
    configs -- the next config starts a fresh process.
    """
    import queue
    import threading
    from tqdm import tqdm

    # Round-robin configs across GPUs.
    per_gpu = [[] for _ in range(num_gpus)]
    cfg_by_label = {}
    for i, cfg in enumerate(configs):
        if cfg.label in hsaco_paths:
            gpu_id = i % num_gpus
            item = (
                cfg.label,
                hsaco_paths[cfg.label],
                cfg.kernel_name,
                cfg.num_workgroups,
                cfg.num_threads,
                cfg.gemm_size[DIM_M],
                cfg.gemm_size[DIM_N],
                cfg.gemm_size[DIM_K],
                num_iterations,
                getattr(cfg, "direct_b", False),
                getattr(cfg, "direct_a", False),
            )
            per_gpu[gpu_id].append(item)
            cfg_by_label[cfg.label] = cfg

    total = sum(len(g) for g in per_gpu)
    results, failed = [], []
    best_tf = 0.0
    pbar = tqdm(total=total, desc=desc, unit="cfg")
    result_q = queue.Queue()

    # One thread per GPU, each processing its queue sequentially.
    threads = []
    for gpu_id in range(num_gpus):
        if not per_gpu[gpu_id]:
            continue
        t = threading.Thread(
            target=_run_gpu_queue,
            args=(gpu_id, per_gpu[gpu_id], result_q),
            daemon=True,
        )
        t.start()
        threads.append(t)

    collected = 0
    try:
        while collected < total:
            try:
                label, times, err = result_q.get(timeout=1.0)
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue
            collected += 1
            cfg = cfg_by_label[label]
            if err or times is None:
                failed.append((cfg, err or "unknown"))
            else:
                post_warmup = times[WARMUP_ITERATIONS:]
                if not post_warmup or min(post_warmup) <= 0:
                    failed.append((cfg, "no valid measurements"))
                    continue
                ns = min(post_warmup)
                tf = cfg.total_flops / ns * 1e-3
                pct = tf / MI300X_PEAK_TFLOPS_F16 * 100
                results.append((cfg, ns / 1e6, tf, pct))
                if tf > best_tf:
                    best_tf = tf
            pbar.update(1)
            pbar.set_postfix_str(f"best {best_tf:.1f} TF, fail={len(failed)}")
    except KeyboardInterrupt:
        remaining = total - collected
        print(
            f"\nCtrl+C -- stopping execution ({collected}/{total} done, "
            f"{remaining} skipped). Collecting partial results..."
        )
    pbar.close()

    alive = [t for t in threads if t.is_alive()]
    if alive:
        import time

        print(
            f"Waiting for {len(alive)} GPU threads to finish current work (timeout 5s)...",
            end="",
            flush=True,
        )
        t0 = time.monotonic()
        for t in alive:
            t.join(timeout=max(0.1, 5.0 - (time.monotonic() - t0)))
        still = sum(1 for t in threads if t.is_alive())
        if still:
            print(f" {still} still busy, moving on.")
        else:
            print(" done.")
    return results, failed


def _verify_gpu_queue(gpu_id, work_queue, result_queue, timeout=180):
    """Process a verification queue on one GPU using a persistent worker."""
    _run_gpu_queue(gpu_id, work_queue, result_queue, timeout=timeout, loop_fn=_persistent_verify_loop)


def verify_on_gpus(configs, hsaco_paths, num_gpus, desc="Verifying"):
    """Verify configs against numpy using persistent workers, all GPUs concurrently.

    Each GPU gets a dedicated thread with a persistent worker that
    reuses HIP context across configs.  On crash the worker is auto-
    respawned.
    """
    import queue
    import threading
    from tqdm import tqdm

    per_gpu = [[] for _ in range(num_gpus)]
    idx = 0
    for cfg in configs:
        if cfg.label not in hsaco_paths:
            continue
        gpu_id = idx % num_gpus
        item = (
            cfg.label,
            hsaco_paths[cfg.label],
            cfg.kernel_name,
            cfg.num_workgroups,
            cfg.num_threads,
            cfg.gemm_size[DIM_M],
            cfg.gemm_size[DIM_N],
            cfg.gemm_size[DIM_K],
            getattr(cfg, "direct_b", False),
            getattr(cfg, "direct_a", False),
        )
        per_gpu[gpu_id].append(item)
        idx += 1

    total = sum(len(g) for g in per_gpu)
    passed, errors = 0, []
    pbar = tqdm(total=total, desc=desc, unit="cfg")
    result_q = queue.Queue()

    threads = []
    for gpu_id in range(num_gpus):
        if not per_gpu[gpu_id]:
            continue
        t = threading.Thread(
            target=_verify_gpu_queue,
            args=(gpu_id, per_gpu[gpu_id], result_q),
            daemon=True,
        )
        t.start()
        threads.append(t)

    collected = 0
    try:
        while collected < total:
            try:
                label, _, err = result_q.get(timeout=1.0)
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue
            collected += 1
            if err:
                errors.append(f"{label}: {err}")
            else:
                passed += 1
            pbar.update(1)
            pbar.set_postfix_str(f"pass={passed}, fail={len(errors)}")
    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping verification, reporting partial results...")
    pbar.close()

    for t in threads:
        t.join(timeout=5.0)
    return passed, errors


def _sweep_summary(results, compile_errs, exec_errs, repro_cmd_fn):
    """Print sweep summary and save result/error files (shared by both sweep modes)."""
    results.sort(key=lambda r: r[2], reverse=True)
    saved_files = []

    if results:
        lines = []
        for i, (c, ms, tf, pct) in enumerate(results):
            line = f"#{i + 1:>3} {tf:>7.1f} TF {pct:>5.1f}% {ms:>8.2f}ms {c.label}"
            if repro_cmd_fn:
                try:
                    line += f" | repro: {repro_cmd_fn(c)}"
                except Exception:
                    pass
            lines.append(line)
        p = _save_tmpfile("bench_results_", lines)
        saved_files.append(p)
        print(f"\nResults ({len(results)}) saved in {p}")
    if compile_errs:
        p = _save_error_file("bench_compile_errors_", "compile", compile_errs, repro_cmd_fn)
        saved_files.append(p)
        print(f"{len(compile_errs)} compile errors in {p}")
    if exec_errs:
        p = _save_error_file("bench_exec_errors_", "exec", exec_errs, repro_cmd_fn)
        saved_files.append(p)
        print(f"{len(exec_errs)} exec errors in {p}")

    print(f"\nSummary: {len(results)} ok, {len(compile_errs)} compile fail, {len(exec_errs)} exec fail")
    if results:
        top_n = min(20, len(results))
        print(f"Top {top_n}:")
        for i, (c, ms, tf, pct) in enumerate(results[:top_n]):
            print(f"  #{i + 1} {c.label}: {tf:.1f} TF ({pct:.1f}%)")
    if saved_files:
        print("\nSaved files:")
        for f in saved_files:
            print(f"  {f}")


def _drain_exec_results(result_q, cfg_by_label, results, exec_failed):
    """Drain completed exec results from a queue.

    Returns (n_drained, best_tf).
    """
    import queue as _qm

    n = 0
    best_tf = max((r[2] for r in results), default=0.0)
    while not result_q.empty():
        try:
            label_r, times, err = result_q.get_nowait()
        except _qm.Empty:
            break
        n += 1
        c = cfg_by_label.get(label_r)
        if not c:
            continue
        if err or times is None:
            exec_failed.append((c, err or "unknown"))
        else:
            post_warmup = times[WARMUP_ITERATIONS:]
            if post_warmup and min(post_warmup) > 0:
                ns = min(post_warmup)
                tf = c.total_flops / ns * 1e-3
                pct = tf / MI300X_PEAK_TFLOPS_F16 * 100
                results.append((c, ns / 1e6, tf, pct))
                if tf > best_tf:
                    best_tf = tf
    return n, best_tf


def bench_perf_sweep_pipelined(
    configs,
    compile_fn,
    repro_cmd_fn,
    *,
    mcpu: str,
    num_gpus=None,
    compile_workers=None,
    compile_timeout=DEFAULT_COMPILE_TIMEOUT,
    post_compile_filter=None,
    zero_init=False,
    iterations=None,
):
    """Pipelined compile+execute: GPU execution starts as HSACOs become available.

    Reuses _run_gpu_queue (with queue.Queue for dynamic item feeding)
    and _exec_one_isolated (with gpu_lock_path for crash-safe GPU
    exclusivity).

    Returns (results, hsaco_paths) -- same interface as
    bench_perf_sweep.
    """
    import multiprocessing as mp
    import queue
    import threading

    from tqdm import tqdm

    if iterations is None:
        iterations = NUM_ITERATIONS
    if num_gpus is None:
        num_gpus = detect_num_gpus(mcpu)
    if compile_workers is None:
        compile_workers = DEFAULT_COMPILE_WORKERS
    check_numpy_blas(label="sweep")

    active = list(configs)

    # Worker budget: +25% extra processes for execution (data prep is CPU-bound).
    if num_gpus > 0:
        n_exec_per_gpu = max(1, max(1, compile_workers // 4) // num_gpus)
    else:
        n_exec_per_gpu = 0

    print(
        f"\nPipelined sweep: {len(active)} configs, "
        f"{compile_workers} compile + {n_exec_per_gpu * num_gpus} exec workers "
        f"({n_exec_per_gpu}/GPU), {num_gpus} GPU(s)"
    )
    sys.stdout.flush()

    spawn_ctx = mp.get_context("spawn")
    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")

    # Per-GPU file locks for crash-safe GPU exclusivity.
    gpu_lock_dir = os.path.join(hsaco_dir, "gpu_locks")
    os.makedirs(gpu_lock_dir)

    # Per-GPU work queues + threads (reuses _run_gpu_queue with Queue mode).
    gpu_work_queues = []
    gpu_threads = []
    exec_result_q = queue.Queue()
    for gpu_id in range(num_gpus):
        lock_path = os.path.join(gpu_lock_dir, f"gpu_{gpu_id}.lock")
        for _ in range(n_exec_per_gpu):
            wq = queue.Queue()
            t = threading.Thread(
                target=_run_gpu_queue,
                args=(gpu_id, wq, exec_result_q),
                kwargs={"gpu_lock_path": lock_path},
                daemon=True,
            )
            t.start()
            gpu_work_queues.append(wq)
            gpu_threads.append(t)

    # Compile pool.
    compile_pool = ProcessPoolExecutor(max_workers=compile_workers, mp_context=spawn_ctx)

    hsaco_paths, resources_map = {}, {}
    compile_failed, exec_failed = [], []
    results = []
    cfg_by_label = {c.label: c for c in active}

    compile_futs = {compile_pool.submit(compile_one, c, hsaco_dir, compile_fn, compile_timeout): c for c in active}
    total_compile = len(active)
    n_compiled, n_exec_submitted, n_exec_done = 0, 0, 0
    exec_rr = 0

    pbar = tqdm(total=total_compile, desc="Compiling", unit="cfg")

    interrupted = False
    try:
        for fut in as_completed(compile_futs):
            cfg = compile_futs[fut]
            n_compiled += 1
            pbar.update(1)
            try:
                label, path, res = fut.result()
                hsaco_paths[label] = path
                if res:
                    resources_map[label] = res

                skip_exec = False
                if post_compile_filter:
                    if not res:
                        skip_exec = True
                        compile_failed.append((cfg, "compile: metadata parse failed", ""))
                    else:
                        reason = post_compile_filter(cfg, res)
                        if reason:
                            skip_exec = True
                            # Categorize on the short prefix (e.g. "occupancy: ...") so
                            # _save_error_file groups identical-kind rejections together.
                            short = reason.split(" -- ", 1)[0]
                            compile_failed.append((cfg, f"compile: {short}", reason))
                if not skip_exec and gpu_work_queues:
                    exec_item = (
                        label,
                        path,
                        cfg.kernel_name,
                        cfg.num_workgroups,
                        cfg.num_threads,
                        cfg.gemm_size[DIM_M],
                        cfg.gemm_size[DIM_N],
                        cfg.gemm_size[DIM_K],
                        iterations,
                        getattr(cfg, "direct_b", False),
                        getattr(cfg, "direct_a", False),
                        zero_init,
                    )
                    gpu_work_queues[exec_rr % len(gpu_work_queues)].put(exec_item)
                    exec_rr += 1
                    n_exec_submitted += 1
            except Exception as e:
                full_err = str(e).strip()
                short = full_err.split("\n")[0].strip()[:200]
                if not short:
                    short = type(e).__name__
                compile_failed.append((cfg, f"compile: {short}", full_err))

            dn, best_tf = _drain_exec_results(exec_result_q, cfg_by_label, results, exec_failed)
            n_exec_done += dn
            n_fail = len(compile_failed) + len(exec_failed)
            pbar.set_postfix_str(
                f"C={n_compiled}/{total_compile} X={n_exec_done}/{n_exec_submitted} best={best_tf:.1f}TF fail={n_fail}"
            )
    except KeyboardInterrupt:
        interrupted = True
        pbar.close()
        pbar = None
        pending = total_compile - n_compiled
        in_flight = n_exec_submitted - n_exec_done
        print(f"\nCtrl+C -- stopping pipeline ({pending} compiles pending, {in_flight} execs in flight)")
        print("Cancelling pending compilations...", end="", flush=True)
        for f in compile_futs:
            f.cancel()
        print(" done.")
    finally:
        compile_pool.shutdown(wait=False, cancel_futures=True)
    if pbar is not None:
        pbar.close()

    # Signal exec threads to stop and wait for in-flight work.
    in_flight = n_exec_submitted - n_exec_done
    for wq in gpu_work_queues:
        wq.put(None)
    if in_flight > 0:
        import time

        drain_timeout = 10.0 if interrupted else 180.0
        t0 = time.monotonic()
        drained = 0
        remaining_count = in_flight
        while any(t.is_alive() for t in gpu_threads):
            elapsed = time.monotonic() - t0
            if elapsed >= drain_timeout:
                break
            print(
                f"\rDraining {remaining_count} in-flight exec results "
                f"(timeout {drain_timeout:.0f}s, {elapsed:.0f}s elapsed)...",
                end="",
                flush=True,
            )
            dn, _ = _drain_exec_results(exec_result_q, cfg_by_label, results, exec_failed)
            drained += dn
            remaining_count = max(0, remaining_count - dn)
            for t in gpu_threads:
                if t.is_alive():
                    t.join(timeout=0.5)
                    break
        dn, _ = _drain_exec_results(exec_result_q, cfg_by_label, results, exec_failed)
        drained += dn
        remaining_count = max(0, remaining_count - dn)
        n_exec_done += drained
        elapsed = time.monotonic() - t0
        still_running = sum(1 for t in gpu_threads if t.is_alive())
        if still_running:
            print(
                f"\rDraining {remaining_count} in-flight exec results -- "
                f"{drained} collected, {still_running} workers still busy "
                f"({elapsed:.1f}s, giving up)."
            )
        else:
            print(f"\rDraining done: {drained} collected ({elapsed:.1f}s)." + " " * 30)
    else:
        for t in gpu_threads:
            t.join(timeout=5.0)

    print(f"Compiled: {len(hsaco_paths)} ok, {len(compile_failed)} failed")
    if n_exec_submitted:
        print(f"Executed: {n_exec_done}/{n_exec_submitted}, {len(exec_failed)} failed")
    elif num_gpus == 0:
        print("No GPUs detected -- execution skipped.")

    compile_errs = list(compile_failed)
    exec_errs = [(c, e, "") for c, e in exec_failed]
    _sweep_summary(results, compile_errs, exec_errs, repro_cmd_fn)
    return results, hsaco_paths


# -- Sweep -----------------------------------------------------------------


def bench_perf_sweep(
    configs,
    compile_fn,
    repro_cmd_fn,
    *,
    mcpu: str,
    num_gpus=None,
    compile_workers=None,
    compile_timeout=DEFAULT_COMPILE_TIMEOUT,
    post_compile_filter=None,
    iterations=None,
):
    """Phase 1 (compile) + Phase 2 (execute).

    Returns (results, hsaco_paths).
    """
    from tqdm import tqdm

    if iterations is None:
        iterations = NUM_ITERATIONS
    if num_gpus is None:
        num_gpus = detect_num_gpus(mcpu)
    if compile_workers is None:
        compile_workers = DEFAULT_COMPILE_WORKERS
    check_numpy_blas(label="sweep")

    active = list(configs)

    print(f"\nCompiling {len(active)} configs, {compile_workers} workers, {num_gpus} GPU(s)")
    sys.stdout.flush()

    # Write manifest so the user can review/edit before compiling.
    manifest_fd, manifest_path = tempfile.mkstemp(prefix="bench_manifest_", suffix=".txt")
    with os.fdopen(manifest_fd, "w") as f:
        for c in active:
            repro = repro_cmd_fn(c) if repro_cmd_fn else c.label
            f.write(f"{c.label}\t{repro}\n")
    print(f"\nManifest: {manifest_path}")
    print("Review/edit the file to remove lines, then press Enter to compile (or Ctrl-C to abort).")
    sys.stdout.flush()
    try:
        input()
    except EOFError:
        pass
    # Re-read manifest: keep only configs whose label is still present.
    with open(manifest_path) as f:
        keep_labels = {line.split("\t")[0] for line in f if line.strip()}
    before = len(active)
    active = [c for c in active if c.label in keep_labels]
    if len(active) < before:
        print(f"Narrowed {before} -> {len(active)} configs from edited manifest")

    # Phase 1: compile.
    import multiprocessing as mp

    hsaco_dir = tempfile.mkdtemp(prefix="bench_hsaco_")
    hsaco_paths, resources_map, failed = {}, {}, []
    spawn_ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=compile_workers, mp_context=spawn_ctx)
    futs = {pool.submit(compile_one, c, hsaco_dir, compile_fn, compile_timeout): c for c in active}
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
        pending = len(futs) - len(hsaco_paths) - len(failed)
        print(
            f"\nCtrl+C -- stopping compilation "
            f"({len(hsaco_paths)} compiled, {pending} cancelled). "
            f"Moving to execution with what we have..."
        )
        for f in futs:
            f.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    pbar.close()
    print(f"Compiled: {len(hsaco_paths)} ok, {len(failed)} failed")

    # Post-compile filter.  Also reject configs where metadata parsing
    # returned None (can't verify occupancy -> unsafe to execute).
    if post_compile_filter:
        before = len(hsaco_paths)
        for c in active:
            if c.label not in hsaco_paths:
                continue
            res = resources_map.get(c.label)
            if not res:
                del hsaco_paths[c.label]
                failed.append((c, "compile: metadata parse failed (no kernel resources)", ""))
                continue
            reason = post_compile_filter(c, res)
            if reason:
                del hsaco_paths[c.label]
                short = reason.split(" -- ", 1)[0]
                failed.append((c, f"compile: {short}", reason))
        dropped = before - len(hsaco_paths)
        if dropped:
            print(f"Post-compile filter: {dropped} skipped (see error file)")

    # Phase 2: execute in subprocesses (crash-isolated).
    exec_active = [c for c in active if c.label in hsaco_paths]

    if num_gpus == 0:
        print("\nNo GPUs detected -- skipping execution phase.")
        results, exec_failed = [], []
    else:
        print(f"\n--- Executing {len(exec_active)} configs ({num_gpus} GPU(s)) ---")
        results, exec_failed = run_on_gpus(
            exec_active,
            hsaco_paths,
            iterations,
            num_gpus,
            desc="Executing",
        )
    failed.extend((c, e, "") for c, e in exec_failed)

    compile_errs = [(c, e, full) for c, e, full in failed if e.startswith("compile:")]
    exec_errs = [(c, e, full) for c, e, full in failed if not e.startswith("compile:")]
    _sweep_summary(results, compile_errs, exec_errs, repro_cmd_fn)

    return results, hsaco_paths


# -- Single-config mode ----------------------------------------------------


def make_inputs(cfg, zero_init=False):
    from kittens.gemm_config import A as OP_A, B as OP_B

    shape_a = cfg.spec.operand_shape(OP_A)
    shape_b = cfg.spec.operand_shape(OP_B)
    if zero_init:
        A = np.zeros(shape_a, dtype=np.float16)
        B = np.zeros(shape_b, dtype=np.float16)
    else:
        np.random.seed(42)
        A = (np.random.randn(*shape_a) * 0.1).astype(np.float16)
        B = (np.random.randn(*shape_b) * 0.1).astype(np.float16)
    return A, B


def print_config(cfg, resources=None, iterations=None):
    gs = cfg.gemm_size
    print(f"Config: {cfg.label}")
    print(f"  problem:    M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}")
    print(
        f"  grid:       {cfg.mapping.num_workgroups_per_kernel} WGs, "
        f"{cfg.mapping.num_waves_per_workgroup} waves/WG, "
        f"{cfg.num_threads} threads"
    )
    print(f"  tiles/WG:   {cfg.mapping.num_tiles_per_workgroup} (per-wave: {cfg.mapping.num_tiles_per_wave})")
    print(f"  pipeline:   strategy={cfg.pipeline_strategy}")
    print(f"  memory:     load_type={cfg.load_type}, b_path={cfg.b_path}, LDS={cfg.lds_bytes} bytes")
    lcm = "lcm" if cfg.lcm_unroll else "no-lcm"
    peel = "peel" if cfg.epilogue_peeling else "no-peel"
    print(f"  unroll:     {lcm}, multiplier={cfg.unroll_factor_multiplier}, {peel}")
    sched = []
    if cfg.ll_sched:
        sched.append("ll-sched")
    if cfg.hoist_wait:
        sched.append("hoist-wait")
    if cfg.num_wg_per_cu > 1:
        sched.append(f"wg_per_cu={cfg.num_wg_per_cu}")
    if sched:
        print(f"  sched:      {', '.join(sched)}")
    iters = iterations if iterations is not None else NUM_ITERATIONS
    print(f"  iterations: {iters} (warmup={WARMUP_ITERATIONS})")
    if resources:
        print(f"  resources:  {resources}")


def run_single(cfg, compile_fn, args, execute_fn):
    from aster.compiler.metadata import (
        parse_asm_kernel_resources,
        compute_register_budget,
    )

    kname = cfg.kernel_name
    iterations = getattr(args, "iterations", None) or NUM_ITERATIONS
    print_ir = getattr(args, "print_ir_after_all", False)
    print_asm = getattr(args, "print_asm", False)

    wg = getattr(args, "num_wg_per_cu", 1) or 1
    agpr_est = getattr(cfg, "estimated_agprs", 0) or 0
    bv, ba, _ = compute_register_budget(
        cfg.num_threads,
        mcpu=cfg.mapping.mcpu,
        num_wg_per_cu=wg,
        agpr_hint=agpr_est,
    )
    nv = getattr(args, "num_vgprs", None) or bv
    na = getattr(args, "num_agprs", None) or ba
    compile_kw = dict(print_ir_after_all=print_ir, print_asm=print_asm, num_vgprs=nv, num_agprs=na)
    print(f"  register budget: vgpr={nv}, agpr={na} (wg_per_cu={wg})")

    if args.compile_only:
        if not args.hsaco:
            raise SystemExit("--compile-only requires --hsaco")
        _, asm = compile_fn(cfg, args.hsaco, **compile_kw)
        res = parse_asm_kernel_resources(asm, kernel_name=kname).get(kname)
        print_config(cfg, res, iterations=iterations)
        if res:
            for v in res.check_occupancy(
                cfg.num_threads, mcpu=cfg.mcpu, num_wg_per_cu=getattr(cfg, "num_wg_per_cu", 1)
            ):
                print(f"  OCCUPANCY ERROR: {v}")
        print(f"  Compiled: {args.hsaco}")
        return

    has_gpu = detect_num_gpus(args.mcpu) > 0

    # Compile (or use pre-compiled HSACO).
    hsaco_path = args.hsaco
    hsaco_tmp = None
    if not hsaco_path:
        import tempfile as _tmp

        hsaco_tmp = _tmp.NamedTemporaryFile(suffix=".hsaco", delete=False)
        hsaco_path = hsaco_tmp.name
        _, asm = compile_fn(cfg, hsaco_path, **compile_kw)
        res = parse_asm_kernel_resources(asm, kernel_name=kname).get(kname)
        print_config(cfg, res, iterations=iterations)
        if res:
            violations = res.check_occupancy(
                cfg.num_threads, mcpu=cfg.mcpu, num_wg_per_cu=getattr(cfg, "num_wg_per_cu", 1)
            )
            for v in violations:
                print(f"  OCCUPANCY ERROR: {v}")
            if violations and not getattr(args, "force", False):
                raise SystemExit(1)
    else:
        print_config(cfg, iterations=iterations)

    if not has_gpu:
        print("No GPUs detected -- skipping execution.")
        return

    try:
        # Timing.
        zero_init = getattr(args, "zero_init", False)
        A, B = make_inputs(cfg, zero_init=zero_init)
        _, times_ns = execute_fn(cfg, hsaco_path, iterations, A, B, skip_gpu_check=True)

        measured = times_ns[WARMUP_ITERATIONS:]
        if not measured:
            print(
                f"\nNo measurements after warmup ({iterations} iterations), "
                f"use a number > {WARMUP_ITERATIONS} e.g. --iterations=100"
            )
            return
        min_ns = min(measured)
        if min_ns <= 0:
            print(f"\nInvalid min timing: {min_ns} ns")
            return
        tf = cfg.total_flops / min_ns * 1e-3
        pct = tf / MI300X_PEAK_TFLOPS_F16 * 100
        print(f"\nMin: {min_ns / 1e6:.2f} ms  {tf:.1f} TFLOPS  ({pct:.1f}% peak)")
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

        # Correctness: verify against numpy reference.
        print("\n--- Correctness check ---")
        A, B = make_inputs(cfg)
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        C_output, _ = execute_fn(cfg, hsaco_path, 1, A, B, skip_gpu_check=True)
        try:
            np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
            print("PASS")
        except AssertionError:
            diff = float(np.max(np.abs(C_output - expected)))
            print(f"FAIL: max_abs_diff={diff:.6g}")
    finally:
        if hsaco_tmp:
            try:
                os.unlink(hsaco_tmp.name)
            except OSError:
                pass


# -- CLI args --------------------------------------------------------------


def add_sweep_cli_args(parser, default_mcpu: str = "gfx942"):
    a = parser.add_argument
    a(
        "--mcpu",
        type=str,
        default=default_mcpu,
        help=f"Target GPU arch for compilation (default: {default_mcpu}). "
        "Independent of the host GPU: set explicitly to cross-compile.",
    )
    a(
        "--compile-only",
        action="store_true",
        help="Compile only, skip execution and verification. Required when "
        "the host has no matching GPU -- without it the bench fails hard "
        "so the user is not silently left with zero measurements.",
    )
    a("--compile-sample", type=int, default=4096, help="Configs to compile (0=all)")
    a("--num-gpus", type=int, default=None, help="GPUs (default: auto)")
    a("--compile-workers", type=int, default=DEFAULT_COMPILE_WORKERS)
    a(
        "--compile-timeout",
        type=int,
        default=DEFAULT_COMPILE_TIMEOUT,
        help=f"Per-kernel compile timeout in seconds (default: {DEFAULT_COMPILE_TIMEOUT})",
    )
    a("--no-reg-filter", action="store_true", help="Disable register estimate filter")
    a(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Number of execution iterations per config (default: {NUM_ITERATIONS})",
    )
    a(
        "--zero-init",
        action="store_true",
        default=False,
        help="Use zero-initialized inputs instead of random. "
        "Isolates power throttling: zeros avoid FP denormal handling "
        "and reduce switching activity, giving a power-neutral baseline.",
    )


def add_single_cli_args(parser, default_mcpu: str = "gfx942"):
    a = parser.add_argument
    a(
        "--mcpu",
        type=str,
        default=default_mcpu,
        help=f"Target GPU arch for compilation (default: {default_mcpu}).",
    )
    a("--hsaco", type=str, default=None, help="Pre-compiled HSACO path")
    a("--compile-only", action="store_true")
    a("--print-ir-after-all", action="store_true")
    a("--print-asm", action="store_true")
    a("--force", action="store_true", help="Run despite occupancy violations")
    a("--num-vgprs", type=int, default=None)
    a("--num-agprs", type=int, default=None)
    a("--num-wg-per-cu", type=int, default=1)
    a(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Number of execution iterations (default: {NUM_ITERATIONS})",
    )
    a(
        "--zero-init",
        action="store_true",
        default=False,
        help="Use zero-initialized inputs (power-neutral baseline)",
    )
