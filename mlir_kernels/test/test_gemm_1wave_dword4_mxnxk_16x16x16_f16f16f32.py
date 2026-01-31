"""Integration test for GEMM 1-wave end-to-end kernel execution."""

import argparse
import pytest

from aster import ir
from aster.pass_pipelines import get_pass_pipeline
from mlir_kernels.common import get_library_paths
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
)


@pytest.mark.parametrize(
    "mlir_filename",
    ["gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir"],
)
@pytest.mark.parametrize("kernel_name", ["test_matmul_kernel"])
@pytest.mark.parametrize(
    # fmt: off
    "m,n,k,m_tile,n_tile,k_tile",
    [
        (16, 16, 16, 16, 16, 16),
        (32, 32, 32, 16, 16, 16),
        (32, 32, 64, 16, 16, 16),
        (128, 128, 64, 16, 16, 16),
        (128, 128, 256, 16, 16, 16),
        (32, 32, 32, 32, 16, 32),
        (32, 32, 64, 16, 32, 16),
        (128, 128, 64, 32, 32, 32),
        (128, 128, 128, 32, 64, 64),
        (128, 128, 128, 64, 32, 64),
        (128, 128, 128, 64, 64, 32),
    ],
    # fmt: on
)
@pytest.mark.parametrize("pass_pipeline_name", ["default", "synchronous", "future"])
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
    pass_pipeline_name: str,
    mcpu: str,
    wavefront_size: int = 64,
):
    """Test GEMM 1-wave kernel execution."""
    pass_pipeline = get_pass_pipeline(pass_pipeline_name)
    try:
        config = GEMMConfig(
            m=m,
            n=n,
            k=k,
            m_tile=m_tile,
            n_tile=n_tile,
            k_tile=k_tile,
            num_waves=1,
            mlir_file=get_mlir_file_path(mlir_filename),
            kernel_name=kernel_name,
            pass_pipeline=pass_pipeline,
            mcpu=mcpu,
            wavefront_size=wavefront_size,
        )
    except ValueError as e:
        pytest.skip(f"Invalid configuration: {e}")

    # Per-block dimensions for 1-wave kernel
    m_per_block = config.m // config.num_blocks_m
    n_per_block = config.n // config.num_blocks_n

    # Create config for per-block execution
    block_config = GEMMConfig(
        m=m_per_block,
        n=n_per_block,
        k=k,
        m_tile=m_tile,
        n_tile=n_tile,
        k_tile=k_tile,
        num_waves=1,
        mlir_file=config.mlir_file,
        kernel_name=kernel_name,
        pass_pipeline=pass_pipeline,
        mcpu=mcpu,
        wavefront_size=wavefront_size,
    )

    print(
        f"M={m_per_block} N={n_per_block} K={k} "
        f"tile=({m_tile}x{n_tile}x{k_tile}) blocks={config.num_workgroups}"
    )

    a_data, b_data, c_data = generate_gemm_data(block_config)
    preprocess = make_gemm_preprocess(block_config)
    verify_fn = make_gemm_verify_fn(block_config)

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
            library_paths=get_library_paths(),
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GEMM 1-wave end-to-end kernel execution"
    )
    add_mnk_args(parser, m_default=16, n_default=16, k_default=32)
    add_tile_args(parser, m_tile_default=16, n_tile_default=16, k_tile_default=32)
    add_gpu_args(
        parser,
        mlir_filename_default="gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir",
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
        pass_pipeline_name="default",
        mcpu=args.mcpu,
    )
