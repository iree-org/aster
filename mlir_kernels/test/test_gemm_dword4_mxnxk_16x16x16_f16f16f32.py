"""Integration test for GEMM multi-wave end-to-end kernel execution."""

import argparse
import os
import pytest

from aster import ir
from aster.pass_pipelines import get_pass_pipeline
from mlir_kernels.kernel_utils import (
    GEMMConfig,
    make_gemm_preprocess,
    make_gemm_verify_fn,
    generate_gemm_data,
)
from mlir_kernels.test.test_utils import (
    get_mlir_file_path,
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
@pytest.mark.parametrize("pass_pipeline_name", ["default", "synchronous"])
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
    pass_pipeline_name: str,
    mcpu: str,
    wavefront_size: int = 64,
):
    """Test GEMM multi-wave kernel execution."""
    pass_pipeline = get_pass_pipeline(pass_pipeline_name)
    try:
        config = GEMMConfig(
            m=m,
            n=n,
            k=k,
            m_tile=m_tile,
            n_tile=n_tile,
            k_tile=k_tile,
            num_waves=num_wavefronts,
            mlir_file=get_mlir_file_path(mlir_filename),
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            mcpu=mcpu,
            wavefront_size=wavefront_size,
        )
    except ValueError as e:
        pytest.skip(f"Invalid configuration: {e}")

    a_data, b_data, c_data = generate_gemm_data(config)
    preprocess = make_gemm_preprocess(config)
    verify_fn = make_gemm_verify_fn(config)

    with ir.Context() as ctx:
        compile_and_run_kernel(
            mlir_file=config.mlir_file,
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            ctx=ctx,
            input_args=[a_data, b_data],
            output_args=[c_data],
            grid_dim=(config.num_workgroups, 1, 1),
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
        pass_pipeline_name="default",
        mcpu=args.mcpu,
    )
