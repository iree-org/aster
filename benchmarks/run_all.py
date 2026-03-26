#!/usr/bin/env python3
"""Unified GEMM benchmark runner.

Runs any combination of IREE, Triton, rocBLAS, and hipBLAS GEMM benchmarks for
a given problem size and prints a comparison table.  Results can optionally be
saved to a JSON file.

Usage examples:
    python run_all.py -m 8192 -n 8192 -k 8192
    python run_all.py -m 4096 -n 4096 -k 4096 --backends triton rocblas
    python run_all.py -m 8192 -n 8192 -k 8192 --output-json results.json
    python run_all.py -m 8192 -n 8192 -k 8192 --backends iree --num-its 20
    python run_all.py -m 4096 -n 4096 -k 4096 --backends iree --print-asm
    python run_all.py -m 4096 -n 4096 -k 4096 --backends triton --print-asm
"""

import sys
import traceback

try:
    from tabulate import tabulate as _tabulate

    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

from common.cli import build_parser, config_from_args, resolve_backends, save_json
from common.gemm_config import GEMMConfig

# ---------------------------------------------------------------------------
# Per-backend run helpers
# ---------------------------------------------------------------------------


def _run_iree(config: GEMMConfig, args) -> dict:
    import torch
    from iree_bench.bench_gemm import IREEBenchmark

    device = torch.device("cuda", torch.cuda.current_device())
    bench = IREEBenchmark(config)
    print("\n=== IREE: compiling... ===", flush=True)
    bench.compile(device, print_mlir=args.print_mlir, print_asm=args.print_asm)
    print("=== IREE: profiling... ===", flush=True)
    return bench.run(num_its=args.num_its, warmup=args.warmup)


def _run_triton(config: GEMMConfig, args) -> dict:
    from triton_bench.bench_gemm import TritonBenchmark

    bench = TritonBenchmark(config)
    print("\n=== Triton: benchmarking... ===", flush=True)
    return bench.run(
        warmup=args.warmup, iters=args.num_its, do_print_asm=args.print_asm
    )


def _run_inductor(config: GEMMConfig, args) -> dict:
    import torch
    from inductor_bench.bench_gemm import InductorBenchmark

    device = torch.device("cuda", torch.cuda.current_device())
    bench = InductorBenchmark(config)
    print("\n=== Inductor: compiling... ===", flush=True)
    bench.compile(device)
    print("=== Inductor: profiling... ===", flush=True)
    return bench.run(num_its=args.num_its, warmup=args.warmup)


def _run_rocblas(config: GEMMConfig, args) -> dict:
    from rocblas.bench_gemm import RocBLASBenchmark

    bench = RocBLASBenchmark(config, executable=args.rocblas_bench)
    print("\n=== rocBLAS: benchmarking... ===", flush=True)
    return bench.run(warmup=args.warmup, iters=args.num_its)


def _run_hipblas(config: GEMMConfig, args) -> dict:
    from hipblas.bench_gemm import HipBLASBenchmark

    bench = HipBLASBenchmark(config, executable=args.hipblas_bench)
    print("\n=== hipBLAS: benchmarking... ===", flush=True)
    return bench.run(warmup=args.warmup, iters=args.num_its)


_RUNNERS = {
    "iree": _run_iree,
    "triton": _run_triton,
    "inductor": _run_inductor,
    "rocblas": _run_rocblas,
    "hipblas": _run_hipblas,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser(
        description=(
            "Unified GEMM benchmark — compares IREE, Triton, rocBLAS, and "
            "hipBLAS on a single problem size."
        )
    )
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Dump backend assembly (AMDGCN ISA for Triton, AMDGCN/IREE asm for IREE). "
        "Only applies to --backends iree and triton.",
    )
    parser.add_argument(
        "--print-mlir",
        action="store_true",
        help="Print the Torch-MLIR IR before IREE compilation. Only applies to --backends iree.",
    )
    parser.add_argument(
        "--rocblas-bench",
        type=str,
        default="rocblas-bench",
        metavar="PATH",
        help="Path to the rocblas-bench executable.",
    )
    parser.add_argument(
        "--hipblas-bench",
        type=str,
        default="hipblaslt-bench",
        metavar="PATH",
        help="Path to the hipblaslt-bench executable.",
    )
    args = parser.parse_args()
    config = config_from_args(args)
    backends = resolve_backends(args)

    print("=" * 60, flush=True)
    print("GEMM Benchmark", flush=True)
    print(
        f"  Shape : ({config.m}, {config.k}) @ ({config.n}, {config.k})^T -> ({config.m}, {config.n})",
        flush=True,
    )
    print(f"  dtype : {config.dtype_name()} -> f32 (output)", flush=True)
    print(f"  FLOPs : {config.flops() / 1e9:.3f} GFLOPs", flush=True)
    print(f"  Backends : {', '.join(backends)}", flush=True)
    print("=" * 60, flush=True)

    results: dict[str, dict] = {}
    for backend in backends:
        runner = _RUNNERS[backend]
        try:
            results[backend] = runner(config, args)
        except Exception:
            print(f"[ERROR] {backend} failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results[backend] = {"ms": float("nan"), "tflops": float("nan")}

    # Summary table.
    print("\n" + "=" * 60, flush=True)
    print("Summary:", flush=True)
    rows = []
    for backend, res in results.items():
        ms = res.get("ms", float("nan"))
        tflops = res.get("tflops", float("nan"))
        extra = res.get("best_config", "")
        rows.append([backend, f"{ms:.4f}", f"{tflops:.3f}", extra])

    headers = ["backend", "ms/iter", "TFLOP/s", "notes"]
    if _HAS_TABULATE:
        print(_tabulate(rows, headers=headers, tablefmt="github"), flush=True)
    else:
        print("  ".join(f"{h:<14}" for h in headers))
        for row in rows:
            print("  ".join(f"{str(v):<14}" for v in row))

    if args.output_json:
        save_json(args.output_json, config, results)


if __name__ == "__main__":
    main()
