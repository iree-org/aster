"""Integration test for GEMM multi-wave end-to-end kernel execution."""

import argparse
import os
import pytest
import numpy as np

from aster import ir
from aster.pass_pipelines import (
    DEFAULT_SROA_PASS_PIPELINE,
    TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
)
from mlir_kernels.gemm_config import validate_gemm_config
from mlir_kernels.test.test_utils import (
    get_mlir_file_path,
    make_gemm_preprocess,
    make_gemm_verify_fn,
    compile_and_run_kernel,
    add_mnk_args,
    add_tile_args,
    add_gpu_args,
    add_wavefront_args,
)

_MLIR_KERNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_library_paths():
    """Get paths to library files for multi-wave GEMM."""
    library_dir = os.path.join(_MLIR_KERNELS_DIR, "library", "common")
    return [
        os.path.join(library_dir, "register-init.mlir"),
        os.path.join(library_dir, "indexing.mlir"),
        os.path.join(library_dir, "copies.mlir"),
    ]


@pytest.mark.parametrize(
    "mlir_filename",
    ["gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir"],
)
@pytest.mark.parametrize("kernel_name", ["test_matmul_kernel"])
@pytest.mark.parametrize(
    # fmt: off
    "m,n,k,m_tile,n_tile,k_tile,num_wavefronts",
    [
        (16, 16, 16, 16, 16, 16, 1),
        (32, 32, 32, 16, 16, 16, 1),
        (32, 32, 64, 16, 16, 16, 1),
        (128, 128, 64, 16, 16, 16, 1),
        (128, 128, 256, 16, 16, 16, 1),
        (32, 32, 32, 32, 16, 32, 1),
        (32, 32, 64, 16, 32, 16, 1),
        (128, 128, 64, 32, 32, 32, 1),
        (128, 128, 256, 32, 32, 16, 1),
        (1024, 1024, 1024, 64, 64, 64, 4),
    ],
    # fmt: on
)
@pytest.mark.parametrize(
    "pass_pipeline", [DEFAULT_SROA_PASS_PIPELINE, TEST_SYNCHRONOUS_SROA_PASS_PIPELINE]
)
@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_gemm_e2e_kernel(
    mlir_filename: str,
    kernel_name: str,
    m: int,
    n: int,
    k: int,
    m_tile: int,
    n_tile: int,
    k_tile: int,
    num_wavefronts: int,
    pass_pipeline: str,
    mcpu: str,
    wavefront_size: int = 64,
):
    """Test GEMM multi-wave kernel execution."""
    is_valid, error = validate_gemm_config(
        m, n, k, m_tile, n_tile, k_tile, num_wavefronts
    )
    if not is_valid:
        pytest.skip(f"Invalid configuration: {error}")

    dt_a, dt_b, dt_c = np.float16, np.float16, np.float32
    size_a = np.dtype(dt_a).itemsize
    size_b = np.dtype(dt_b).itemsize

    num_blocks_m = m // m_tile
    num_blocks_n = n // n_tile
    num_blocks = num_blocks_m * num_blocks_n
    num_threads = 64 * num_wavefronts
    # Per-block dimensions
    m_per_block = m // num_blocks_m
    n_per_block = n // num_blocks_n
    print(
        f"M={m_per_block} N={n_per_block} K={k} tile=({m_tile}x{n_tile}x{k_tile}) blocks={num_blocks}"
    )

    preprocess = make_gemm_preprocess(
        m=m_per_block,
        n=n_per_block,
        k=k,
        m_tile=m_tile,
        n_tile=n_tile,
        k_tile=k_tile,
        num_blocks=num_blocks,
        num_threads=num_threads,
        size_a=size_a,
        size_b=size_b,
        num_wavefronts=num_wavefronts,
    )

    a_data = np.random.randn(m, k).astype(dt_a)
    b_data = np.random.randn(k, n).astype(dt_b)
    c_data = np.zeros((m * n), dtype=dt_c)

    verify_fn = make_gemm_verify_fn(m, n, k, dt_a, dt_b, dt_c)

    with ir.Context() as ctx:
        compile_and_run_kernel(
            mlir_file=get_mlir_file_path(mlir_filename),
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            ctx=ctx,
            input_args=[a_data, b_data],
            output_args=[c_data],
            grid_dim=(num_blocks, 1, 1),
            block_dim=(num_threads, 1, 1),
            verify_fn=verify_fn,
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            preprocess=preprocess,
            library_paths=_get_library_paths(),
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GEMM multi-wave end-to-end kernel execution"
    )
    add_mnk_args(parser, m_default=16, n_default=16, k_default=32)
    add_tile_args(parser, m_tile_default=16, n_tile_default=16, k_tile_default=32)
    add_wavefront_args(parser, num_wavefronts_default=1)
    add_gpu_args(
        parser,
        mlir_filename_default="gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir",
    )
    args = parser.parse_args()

    test_gemm_e2e_kernel(
        mlir_filename=args.mlir_filename,
        kernel_name="test_matmul_kernel",
        m=args.m,
        n=args.n,
        k=args.k,
        m_tile=args.m_tile,
        n_tile=args.n_tile,
        k_tile=args.k_tile,
        num_wavefronts=args.num_wavefronts,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=args.mcpu,
    )
