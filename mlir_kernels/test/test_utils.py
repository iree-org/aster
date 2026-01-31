"""Common utilities for mlir_kernels tests."""

import argparse
import os
from typing import Callable, Optional, List, Tuple
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)

# MFMA operation dimensions (16x16x16)
MFMA_SIZE_M = 16
MFMA_SIZE_N = 16
MFMA_SIZE_K = 16
MFMA_SIZE = MFMA_SIZE_M * MFMA_SIZE_N * MFMA_SIZE_K

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR_KERNELS_DIR = os.path.dirname(_TEST_DIR)


def get_mlir_file_path(mlir_filename: str) -> str:
    """Get full path to an MLIR file in mlir_kernels/."""
    return os.path.join(_MLIR_KERNELS_DIR, mlir_filename)


# ============================================================================
# CLI Argument Helpers
# ============================================================================


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


# ============================================================================
# GEMM Preprocess Helpers
# ============================================================================


def make_gemm_preprocess(
    m: int,
    n: int,
    k: int,
    m_tile: int,
    n_tile: int,
    k_tile: int,
    num_blocks: int,
    num_threads: int,
    size_a: int,
    size_b: int,
    num_wavefronts: int = 1,
) -> Callable[[str], str]:
    """Create a preprocess function for GEMM MLIR templates.

    Args:
        m, n, k: Problem dimensions (per-block for multi-block)
        m_tile, n_tile, k_tile: Tile sizes
        num_blocks: Number of workgroups
        num_threads: Threads per workgroup
        size_a, size_b: Bytes per element for A and B
        num_wavefronts: Number of wavefronts per workgroup
    """

    def preprocess(x: str) -> str:
        x = x.replace("{{SIZE_M}}", str(m))
        x = x.replace("{{SIZE_N}}", str(n))
        x = x.replace("{{SIZE_K}}", str(k))
        x = x.replace("{{TILE_SIZE_M}}", str(m_tile))
        x = x.replace("{{TILE_SIZE_N}}", str(n_tile))
        x = x.replace("{{TILE_SIZE_K}}", str(k_tile))
        x = x.replace("{{NUM_BLOCKS}}", str(num_blocks))
        x = x.replace("{{NUM_THREADS}}", str(num_threads))
        x = x.replace(
            "{{LDS_SIZE}}",
            str(m_tile * k_tile * size_a + n_tile * k_tile * size_b),
        )
        size_k_by_tile = k // k_tile
        x = x.replace("{{SIZE_K_BY_TILE_SIZE_K}}", str(size_k_by_tile))

        # MFMA loop count
        mnkt = m_tile * n_tile * k_tile
        mnkt_mfma = mnkt // MFMA_SIZE
        loop_size = mnkt_mfma // num_wavefronts
        assert mnkt % MFMA_SIZE == 0, "Invalid configuration"
        assert mnkt_mfma % num_wavefronts == 0, "Invalid configuration"
        x = x.replace("{{LOOP_SIZE_D_MMNNKK}}", str(loop_size))
        return x

    return preprocess


# ============================================================================
# Verification Helpers
# ============================================================================


def make_gemm_verify_fn(
    m: int,
    n: int,
    k: int,
    dt_a=np.float16,
    dt_b=np.float16,
    dt_c=np.float32,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Callable:
    """Create a verification function for GEMM kernels.

    Expects B in transposed layout (n, k) and output C as flat (m*n,).
    """

    def verify_fn(input_args, output_args):
        a_flat = np.array(input_args[0])
        a = a_flat.reshape(m, k)
        b_flat = np.array(input_args[1])
        b = b_flat.reshape(n, k).T  # B stored as (n, k), need (k, n)
        c_flat = np.array(output_args[0], dtype=dt_c)
        c = c_flat.reshape(m, n)

        ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        rel_error = np.linalg.norm(c - ref) / np.linalg.norm(ref)
        print(f"Error: {rel_error}")

        diff = np.abs(c.astype(np.float32) - ref)
        diff[np.where(diff < 1e-5)] = 0.0
        assert np.allclose(c, ref, rtol=rtol, atol=atol), (
            f"GEMM kernel failed!\n"
            f"Max diff: {np.max(np.abs(c - ref))}\n"
            f"c shape: {c.shape}, ref shape: {ref.shape}\n"
            f"diff:\n{np.array2string(diff, precision=4, suppress_small=True)}"
        )

    return verify_fn


# ============================================================================
# Kernel Execution Helpers
# ============================================================================


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
