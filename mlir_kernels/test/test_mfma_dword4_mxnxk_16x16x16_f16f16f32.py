"""Integration test for MFMA end-to-end kernel execution."""

import argparse
import os
from typing import Callable
import pytest
import numpy as np

from aster import ir
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE
from mlir_kernels.test.test_utils import (
    MFMA_SIZE_M,
    MFMA_SIZE_N,
    MFMA_SIZE_K,
    get_mlir_file_path,
    compile_and_run_kernel,
    add_mnk_args,
    add_gpu_args,
)

_MLIR_KERNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE_NAME = "mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir"
KERNEL_NAME = "test_matmul_kernel"


def _get_library_paths():
    """Get paths to library files for MFMA test."""
    return [os.path.join(_MLIR_KERNELS_DIR, "library", "common", "indexing.mlir")]


def _make_mfma_preprocess(
    m: int, n: int, k: int, size_a: int, size_b: int
) -> Callable[[str], str]:
    """Create preprocess function for MFMA block-based kernels.

    Args:
        m, n, k: Number of 16x16 blocks in each dimension
        size_a, size_b: Bytes per element
    """

    def preprocess(x: str) -> str:
        x = x.replace("{{SIZE_M}}", str(m))
        x = x.replace("{{SIZE_N}}", str(n))
        x = x.replace("{{SIZE_K}}", str(k))
        x = x.replace(
            "{{LDS_B_SHIFT}}",
            str(m * k * MFMA_SIZE_M * MFMA_SIZE_K * size_a),
        )
        x = x.replace(
            "{{LDS_SIZE}}",
            str(
                m * k * MFMA_SIZE_M * MFMA_SIZE_K * size_a
                + k * n * MFMA_SIZE_K * MFMA_SIZE_N * size_b
            ),
        )
        return x

    return preprocess


def _make_mfma_verify_fn(
    batch: int,
    m: int,
    n: int,
    k: int,
    dt_a=np.float16,
    dt_b=np.float16,
    dt_c=np.float32,
) -> Callable:
    """Create verification function for MFMA block-based kernels.

    Args:
        batch: num_workgroups * num_waves
        m, n, k: Number of 16x16 blocks in each dimension
    """

    def verify_fn(input_args, output_args):
        a_flat = np.array(input_args[0])
        a_blocks = a_flat.reshape(batch, m, k, MFMA_SIZE_M, MFMA_SIZE_K)

        b_flat = np.array(input_args[1])
        b_blocks = b_flat.reshape(batch, k, n, MFMA_SIZE_K, MFMA_SIZE_N)

        c_flat = np.array(output_args[0])
        c_blocks = c_flat.reshape(batch, m, n, MFMA_SIZE_M, MFMA_SIZE_N)

        # Compute reference using block matrix multiplication
        ref = np.zeros((batch, m, n, MFMA_SIZE_M, MFMA_SIZE_N), dtype=dt_c)
        for b in range(batch):
            for i in range(m):
                for j in range(n):
                    for l in range(k):
                        a_block = a_blocks[b, i, l]
                        b_block = b_blocks[b, l, j]
                        ref[b, i, j] += np.matmul(
                            a_block.astype(dt_c), b_block.astype(dt_c)
                        )

        if not np.allclose(c_blocks, ref, rtol=1e-5, atol=1e-5):
            diff = np.abs(c_blocks - ref)
            max_diff = np.max(diff)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"MFMA kernel failed! Max diff: {max_diff} at index {max_idx}\n"
                f"c shape: {c_blocks.shape}, ref shape: {ref.shape}\n"
                f"c_blocks:\n{c_blocks}\nref:\n{ref}"
            )

    return verify_fn


@pytest.mark.parametrize(
    # fmt: off
    "mlir_filename,kernel_name,num_workgroups,num_waves,m,n,k,pass_pipeline",
    [
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 1, 1, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 1, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 1, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 2, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 2, 1, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 2, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 3, 3, 3, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 4, 4, 4, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 1, 4, 4, 6, DEFAULT_SROA_PASS_PIPELINE),
        # Test with multiple workgroups and waves
        (FILE_NAME, KERNEL_NAME, 2, 1, 1, 1, 1, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 2, 1, 2, 2, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 2, 1, 1, 1, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 1, 2, 2, 2, 2, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 2, 2, 1, 1, 1, DEFAULT_SROA_PASS_PIPELINE),
        (FILE_NAME, KERNEL_NAME, 2, 2, 2, 2, 2, DEFAULT_SROA_PASS_PIPELINE),
    ],
    # fmt: on
)
@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_mfma_e2e_kernel(
    mlir_filename: str,
    kernel_name: str,
    num_workgroups: int,
    num_waves: int,
    m: int,
    n: int,
    k: int,
    pass_pipeline: str,
    mcpu: str,
    wavefront_size: int = 64,
):
    """Test MFMA end-to-end kernel execution.

    Tests block matrix multiplication where:
    - m, n, k are the number of 16x16 blocks in each dimension
    - Each workgroup/wave needs its own data (batch = num_workgroups * num_waves)
    """
    dt_a, dt_b, dt_c = np.float16, np.float16, np.float32
    size_a = np.dtype(dt_a).itemsize
    size_b = np.dtype(dt_b).itemsize

    batch = num_workgroups * num_waves
    a_size = batch * (m * k) * (MFMA_SIZE_M * MFMA_SIZE_K)
    b_size = batch * (k * n) * (MFMA_SIZE_K * MFMA_SIZE_N)
    c_size = batch * (m * n) * (MFMA_SIZE_M * MFMA_SIZE_N)

    a_data = np.full(a_size, 1.0, dtype=dt_a)
    b_data = np.full(b_size, 2.0, dtype=dt_b)
    c_data = np.zeros(c_size, dtype=dt_c)

    preprocess = _make_mfma_preprocess(m, n, k, size_a, size_b)
    verify_fn = _make_mfma_verify_fn(batch, m, n, k, dt_a, dt_b, dt_c)

    num_threads = num_waves * wavefront_size

    with ir.Context() as ctx:
        compile_and_run_kernel(
            mlir_file=get_mlir_file_path(mlir_filename),
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            ctx=ctx,
            input_args=[a_data, b_data],
            output_args=[c_data],
            grid_dim=(num_workgroups, 1, 1),
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
        description="Test MFMA end-to-end kernel execution with block matrix multiplication"
    )
    add_mnk_args(
        parser,
        m_default=4,
        n_default=4,
        k_default=4,
        m_help="Number of 16x16 blocks in M dimension",
        n_help="Number of 16x16 blocks in N dimension",
        k_help="Number of 16x16 blocks in K dimension",
    )
    add_gpu_args(
        parser,
        mlir_filename_default="mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir",
    )
    parser.add_argument(
        "--num-workgroups",
        type=int,
        default=1,
        help="Number of workgroups (default: 1)",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        default=1,
        help="Number of waves per workgroup (default: 1)",
    )
    args = parser.parse_args()

    test_mfma_e2e_kernel(
        mlir_filename=args.mlir_filename,
        kernel_name=KERNEL_NAME,
        m=args.m,
        n=args.n,
        k=args.k,
        num_workgroups=args.num_workgroups,
        num_waves=args.num_waves,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=args.mcpu,
    )
