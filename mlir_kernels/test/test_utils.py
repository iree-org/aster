"""Test utilities for mlir_kernels tests.

This module provides CLI argument helpers and re-exports compile_and_run from
aster.testing. Shared config classes, preprocess, and verify functions are in
mlir_kernels.kernel_utils.
"""

import argparse
import os
from typing import Callable, Optional, List, Tuple
import numpy as np

from aster import ir, utils
from aster.testing import (
    compile_mlir_file_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)

# Re-export from kernel_utils for convenience
from mlir_kernels.kernel_utils import (
    MFMA_SIZE_M,
    MFMA_SIZE_N,
    MFMA_SIZE_K,
    MFMA_SIZE,
    GEMMConfig,
    BatchedSmallGEMMConfig,
    make_gemm_preprocess,
    make_batchedsmallgemm_preprocess,
    make_gemm_verify_fn,
    make_batchedsmallgemm_verify_fn,
    generate_gemm_data,
    generate_batchedsmallgemm_data,
)

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


# =============================================================================
# Kernel Execution
# =============================================================================


def compile_and_run_kernel(
    mlir_file: str,
    kernel_name: str,
    pass_pipeline: str,
    ctx: ir.Context,
    input_args: List[np.ndarray],
    output_args: List[np.ndarray],
    grid_dim: Tuple[int, int, int],
    block_dim: Tuple[int, int, int],
    verify_fn: Callable,
    mcpu: str = "gfx942",
    wavefront_size: int = 64,
    preprocess: Optional[Callable[[str], str]] = None,
    library_paths: Optional[List[str]] = None,
    print_timings: bool = False,
    print_ir_after_all: bool = False,
    num_iterations: int = 5,
    skip_on_cross_compile: bool = False,
) -> Optional[List[int]]:
    """Compile MLIR to hsaco, execute, and verify.

    Returns iteration times in nanoseconds, or None if skipped.
    """
    import pytest

    asm_complete, module_after_passes = compile_mlir_file_to_asm(
        mlir_file,
        kernel_name,
        pass_pipeline,
        ctx,
        preprocess=preprocess,
        library_paths=library_paths or [],
        print_timings=print_timings,
        print_ir_after_all=print_ir_after_all,
    )

    hsaco_path = utils.assemble_to_hsaco(
        asm_complete, target=mcpu, wavefront_size=wavefront_size
    )
    assert hsaco_path is not None, "Failed to assemble kernel to HSACO"

    with hsaco_file(hsaco_path):
        if not utils.system_has_mcpu(mcpu=mcpu):
            if skip_on_cross_compile:
                print(module_after_passes)
                print(asm_complete)
            pytest.skip(
                f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
            )

        iteration_times = execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name=kernel_name,
            input_args=input_args,
            output_args=output_args,
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            verify_fn=verify_fn,
            grid_dim=grid_dim,
            block_dim=block_dim,
            num_iterations=num_iterations,
        )
        print(f"Iteration times: {iteration_times} nanoseconds")
        return iteration_times
