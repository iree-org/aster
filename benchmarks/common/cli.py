"""Shared CLI argument parsing and JSON output utilities for GEMM benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Any

from .gemm_config import DTYPE_NAMES, GEMMConfig

ALL_BACKENDS = ["iree", "triton", "inductor", "rocblas", "hipblas"]


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser populated with all common benchmark options.

    Individual benchmark scripts may extend the returned parser with their own arguments
    before calling parse_args().
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem size.
    parser.add_argument("-m", type=int, required=True, help="Rows of A / rows of C.")
    parser.add_argument(
        "-n",
        type=int,
        required=True,
        help="Rows of B (stored transposed) / columns of C.",
    )
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Inner dimension (columns of A = columns of B).",
    )

    # Data type.
    parser.add_argument(
        "--dtype",
        "-t",
        type=str,
        default="f16",
        choices=list(DTYPE_NAMES.keys()),
        help="Element type for A and B.",
    )

    # Timing parameters.
    parser.add_argument(
        "--num-its",
        "--iters",
        "-i",
        type=int,
        default=10,
        dest="num_its",
        metavar="N",
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        metavar="N",
        help="Number of warm-up iterations (not measured).",
    )

    # Reproducibility.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for matrix initialisation.",
    )

    # Backend selection.
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=ALL_BACKENDS + ["all"],
        default=["all"],
        metavar="BACKEND",
        help=(
            "Backends to benchmark. Choose from: " + ", ".join(ALL_BACKENDS) + ", all."
        ),
    )

    # Output.
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write benchmark results to this JSON file.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> GEMMConfig:
    """Build a GEMMConfig from the parsed command-line arguments."""
    return GEMMConfig(
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=DTYPE_NAMES[args.dtype],
        transpose_b=True,
        seed=args.seed,
    )


def resolve_backends(args: argparse.Namespace) -> list[str]:
    """Expand 'all' in --backends to the full list."""
    if "all" in args.backends:
        return list(ALL_BACKENDS)
    return list(args.backends)


def save_json(
    path: str,
    config: GEMMConfig,
    results: dict[str, Any],
) -> None:
    """Write *results* and the *config* description to a JSON file at *path*."""
    payload = {
        "config": config.to_dict(),
        "results": results,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Results written to {out}", flush=True)
