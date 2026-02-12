"""Test utilities for mlir_kernels tests.

This module provides:
- get_mlir_file_path: resolve MLIR filenames relative to mlir_kernels/
- CLI argument helpers (add_mnk_args, add_tile_args, add_gpu_args, add_wavefront_args)
"""

import argparse
import os
from typing import Optional

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR_KERNELS_DIR = os.path.dirname(_TEST_DIR)


def get_mlir_file_path(mlir_filename: str) -> str:
    """Get full path to an MLIR file in mlir_kernels/."""
    return os.path.join(_MLIR_KERNELS_DIR, mlir_filename)


# =============================================================================
# CLI Argument Helpers
# =============================================================================


def add_mnk_args(
    parser: argparse.ArgumentParser,
    m_default: int = 16,
    n_default: int = 16,
    k_default: int = 32,
    m_help: str = "Size in M dimension",
    n_help: str = "Size in N dimension",
    k_help: str = "Size in K dimension",
) -> None:
    """Add -m, -n, -k arguments to parser."""
    parser.add_argument("-m", "--m", type=int, default=m_default, help=m_help)
    parser.add_argument("-n", "--n", type=int, default=n_default, help=n_help)
    parser.add_argument("-k", "--k", type=int, default=k_default, help=k_help)


def add_tile_args(
    parser: argparse.ArgumentParser,
    m_tile_default: int = 16,
    n_tile_default: int = 16,
    k_tile_default: int = 32,
) -> None:
    """Add -M, -N, -K tile size arguments to parser."""
    parser.add_argument(
        "-M",
        "--m-tile",
        type=int,
        default=m_tile_default,
        help=f"Tile size in M dimension (default: {m_tile_default})",
    )
    parser.add_argument(
        "-N",
        "--n-tile",
        type=int,
        default=n_tile_default,
        help=f"Tile size in N dimension (default: {n_tile_default})",
    )
    parser.add_argument(
        "-K",
        "--k-tile",
        type=int,
        default=k_tile_default,
        help=f"Tile size in K dimension (default: {k_tile_default})",
    )


def add_gpu_args(
    parser: argparse.ArgumentParser,
    mcpu_default: str = "gfx942",
    mlir_filename_default: Optional[str] = None,
) -> None:
    """Add --mcpu and --mlir-filename arguments to parser."""
    parser.add_argument(
        "--mcpu",
        type=str,
        default=mcpu_default,
        help=f"Target GPU architecture (default: {mcpu_default})",
    )
    if mlir_filename_default:
        parser.add_argument(
            "--mlir-filename",
            type=str,
            default=mlir_filename_default,
            help=f"MLIR filename to test (default: {mlir_filename_default})",
        )


def add_wavefront_args(
    parser: argparse.ArgumentParser,
    num_wavefronts_default: int = 1,
) -> None:
    """Add -W/--num-wavefronts argument to parser."""
    parser.add_argument(
        "-W",
        "--num-wavefronts",
        type=int,
        default=num_wavefronts_default,
        help=f"Number of wavefronts (default: {num_wavefronts_default})",
    )
