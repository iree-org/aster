"""Integration test for BatchedSmallGEMM end-to-end kernel execution."""

import argparse
import os
import pytest

from aster import ir
from aster.pass_pipelines import get_pass_pipeline
from mlir_kernels.kernel_utils import (
    BatchedSmallGEMMConfig,
    make_batchedsmallgemm_preprocess,
    make_batchedsmallgemm_verify_fn,
    generate_batchedsmallgemm_data,
)
from aster.testing import compile_and_run
from mlir_kernels.test.test_utils import (
    get_mlir_file_path,
    add_mnk_args,
    add_gpu_args,
)

_MLIR_KERNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE_NAME = "batchedsmallgemm_dword4_mxnxk_16x16x16_f16f16f32.mlir"
KERNEL_NAME = "test_matmul_kernel"


def _get_library_paths():
    """Get paths to library files for BatchedSmallGEMM test."""
    return [os.path.join(_MLIR_KERNELS_DIR, "library", "common", "indexing.mlir")]


@pytest.mark.parametrize(
    # fmt: off
    "mlir_filename,kernel_name,num_workgroups,num_waves,m,n,k,pass_pipeline_name",
    [
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 1, 1, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 1, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 1, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 1, 2, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 2, 1, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 2, 2, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 3, 3, 3, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 4, 4, 4, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 1, 4, 4, 6, "default"),
        # Test with multiple workgroups and waves
        (FILE_NAME, KERNEL_NAME, 2, 1, 1, 1, 1, "default"),
        (FILE_NAME, KERNEL_NAME, 2, 1, 2, 2, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 2, 1, 1, 1, "default"),
        (FILE_NAME, KERNEL_NAME, 1, 2, 2, 2, 2, "default"),
        (FILE_NAME, KERNEL_NAME, 2, 2, 1, 1, 1, "default"),
        (FILE_NAME, KERNEL_NAME, 2, 2, 2, 2, 2, "default"),
    ],
    # fmt: on
)
@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_batchedsmallgemm_e2e_kernel(
    mlir_filename: str,
    kernel_name: str,
    num_workgroups: int,
    num_waves: int,
    m: int,
    n: int,
    k: int,
    pass_pipeline_name: str,
    mcpu: str,
    wavefront_size: int = 64,
):
    """Test BatchedSmallGEMM end-to-end kernel execution.

    Tests block matrix multiplication where:
    - m, n, k are the number of 16x16 blocks in each dimension
    - Each workgroup/wave needs its own data (batch = num_workgroups * num_waves)
    """
    pass_pipeline = get_pass_pipeline(pass_pipeline_name)
    config = BatchedSmallGEMMConfig(
        m=m,
        n=n,
        k=k,
        num_workgroups=num_workgroups,
        num_waves=num_waves,
        mlir_file=get_mlir_file_path(mlir_filename),
        kernel_name=kernel_name,
        pass_pipeline=pass_pipeline,
        mcpu=mcpu,
        wavefront_size=wavefront_size,
    )

    a_data, b_data, c_data = generate_batchedsmallgemm_data(config)
    preprocess = make_batchedsmallgemm_preprocess(config)
    verify_fn = make_batchedsmallgemm_verify_fn(config)

    with ir.Context() as ctx:
        compile_and_run(
            file_name=config.mlir_file,
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            ctx=ctx,
            input_data=[a_data, b_data],
            output_data=[c_data],
            grid_dim=(num_workgroups, 1, 1),
            block_dim=(config.num_threads, 1, 1),
            verify_fn=verify_fn,
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            preprocess=preprocess,
            library_paths=_get_library_paths(),
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test BatchedSmallGEMM end-to-end kernel execution with block matrix multiplication"
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
        mlir_filename_default="batchedsmallgemm_dword4_mxnxk_16x16x16_f16f16f32.mlir",
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

    test_batchedsmallgemm_e2e_kernel(
        mlir_filename=args.mlir_filename,
        kernel_name=KERNEL_NAME,
        m=args.m,
        n=args.n,
        k=args.k,
        num_workgroups=args.num_workgroups,
        num_waves=args.num_waves,
        pass_pipeline_name="default",
        mcpu=args.mcpu,
    )
