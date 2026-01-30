"""Integration test for MFMA end-to-end kernel execution."""

import argparse
import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)
from aster.pass_pipelines import (
    DEFAULT_SROA_PASS_PIPELINE,
    SYNCHRONOUS_SROA_PASS_PIPELINE,
)
from mlir_kernels.gemm_config import validate_gemm_config


# Block sizes for each MFMA operation dimension (16x16x16)
MFMA_SIZE_M = 16
MFMA_SIZE_N = 16
MFMA_SIZE_K = 16

FILE_NAME = "gemm_dword4_mxnxk_16x16x16_f16f16f32.mlir"
KERNEL_NAME = "test_matmul_kernel"


@pytest.mark.parametrize(
    "mlir_filename",
    [
        # "gemm_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        "gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir",
    ],
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
    "pass_pipeline", [DEFAULT_SROA_PASS_PIPELINE, SYNCHRONOUS_SROA_PASS_PIPELINE]
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
    wavefront_size=64,
):
    """Tes."""

    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(test_dir, "..", mlir_filename)
    register_init_lib = os.path.join(
        test_dir, "..", "library", "common", "register-init.mlir"
    )
    indexing_lib = os.path.join(test_dir, "..", "library", "common", "indexing.mlir")
    copies_lib = os.path.join(test_dir, "..", "library", "common", "copies.mlir")

    # Validate configuration using shared validation logic
    is_valid, error = validate_gemm_config(
        m, n, k, m_tile, n_tile, k_tile, num_wavefronts
    )
    if not is_valid:
        pytest.skip(f"Invalid configuration: {error}")

    with ir.Context() as ctx:
        dt_a, dt_b, dt_c = np.float16, np.float16, np.float32
        size_a = np.dtype(dt_a).itemsize  # bytes per element for A
        size_b = np.dtype(dt_b).itemsize  # bytes per element for B

        num_blocks_m = m // m_tile
        num_blocks_n = n // n_tile
        num_blocks = num_blocks_m * num_blocks_n
        num_threads = 64 * num_wavefronts

        def preprocess(x):
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
                str((m_tile * k_tile * size_a + n_tile * k_tile * size_b)),
            )
            SIZE_K_BY_TILE_SIZE_K = k // k_tile
            x = x.replace("{{SIZE_K_BY_TILE_SIZE_K}}", str(SIZE_K_BY_TILE_SIZE_K))
            # Perform replacement for LOOP_SIZE_D_MMNNKK (proper count needed for
            # scheduling to kick in).
            mnkt = m_tile * n_tile * k_tile
            mnk_mfma = MFMA_SIZE_M * MFMA_SIZE_N * MFMA_SIZE_K
            mnkt_mfma = mnkt // mnk_mfma
            LOOP_SIZE_D_MMNNKK = mnkt_mfma // num_wavefronts
            # These should have been checked by validate_gemm_config; this is a
            # sanity check.
            assert mnkt % mnk_mfma == 0, "Invalid configuration"
            assert mnkt_mfma % num_wavefronts == 0, "Invalid configuration"
            x = x.replace("{{LOOP_SIZE_D_MMNNKK}}", str(LOOP_SIZE_D_MMNNKK))
            return x

        asm_complete, module_after_passes = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            preprocess=preprocess,
            print_ir_after_all=False,
            library_paths=[register_init_lib, indexing_lib, copies_lib],
            print_timings=False,
        )
        # print(asm_complete, flush=True)
        a_data = np.random.randn(m, k).astype(dt_a)
        b_data = np.random.randn(k, n).astype(dt_b)
        c_data = np.zeros((m * n), dtype=dt_c)

        def verify_fn(input_args, output_args):
            a_flat = np.array(input_args[0])
            a = a_flat.reshape(m, k)
            b_flat = np.array(input_args[1])
            b = b_flat.reshape(n, k).T
            c_flat = np.array(output_args[0], dtype=dt_c)
            c = c_flat.reshape(m, n)
            ref_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))
            print(f"Error: {np.linalg.norm(c - ref_f32) / np.linalg.norm(ref_f32)}")
            diff = np.abs(c.astype(np.float32) - ref_f32)
            diff[np.where(diff < 1e-5)] = 0.0
            assert np.allclose(c, ref_f32, rtol=1e-4, atol=1e-4), (
                f"MFMA kernel failed!\n"
                f"Max diff: {np.max(np.abs(c - ref_f32))}\n"
                f"c shape: {c.shape}, ref shape: {ref_f32.shape}\n"
                f"diff:\n{np.array2string(diff, precision=4, suppress_small=True)}"
            )

        # Assemble to hsaco
        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=mcpu, wavefront_size=wavefront_size
        )
        assert hsaco_path is not None, "Failed to assemble kernel to HSACO"

        with hsaco_file(hsaco_path):
            # Skip execution if GPU doesn't match
            if not utils.system_has_mcpu(mcpu=mcpu):
                print(module_after_passes)
                print(asm_complete)
                pytest.skip(
                    f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
                )

            iteration_times = execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=[a_data, b_data],
                output_args=[c_data],
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                verify_fn=verify_fn,
                grid_dim=(num_blocks, 1, 1),
                block_dim=(num_threads, 1, 1),
                num_iterations=5,
            )
            print(f"Iteration times: {iteration_times} nanoseconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MFMA end-to-end kernel execution with block matrix multiplication"
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        default=16,
        help="Number of blocks in M dimension (default: 16)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=16,
        help="Number of blocks in N dimension (default: 16)",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=32,
        help="Number of blocks in K dimension (default: 32)",
    )
    parser.add_argument(
        "-M",
        "--m-tile",
        type=int,
        default=16,
        help="Tile size in M dimension (default: 16)",
    )
    parser.add_argument(
        "-N",
        "--n-tile",
        type=int,
        default=16,
        help="Tile size in N dimension (default: 16)",
    )
    parser.add_argument(
        "-K",
        "--k-tile",
        type=int,
        default=32,
        help="Tile size in K dimension (default: 16)",
    )
    parser.add_argument(
        "-W",
        "--num-wavefronts",
        type=int,
        default=1,
        help="Number of wavefronts (default: 1)",
    )
    parser.add_argument(
        "--mcpu",
        type=str,
        default="gfx942",
        help="Target GPU architecture (default: gfx942)",
    )
    parser.add_argument(
        "--mlir-filename",
        type=str,
        # default="gemm_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        default="gemm_sched_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        help="MLIR filename to test (default: mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir)",
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
        # pass_pipeline=SYNCHRONOUS_SROA_PASS_PIPELINE,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=args.mcpu,
    )
