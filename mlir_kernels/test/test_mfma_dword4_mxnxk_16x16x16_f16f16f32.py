"""Integration test for MFMA end-to-end kernel execution."""

import argparse
import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    DEFAULT_SROA_PASS_PIPELINE,
    hsaco_file,
)

# Block sizes for each MFMA operation dimension (16x16x16)
M_BLOCK_SIZE = 16
N_BLOCK_SIZE = 16
K_BLOCK_SIZE = 16

FILE_NAME = "mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir"
KERNEL_NAME = "test_matmul_kernel"


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
    mlir_filename,
    kernel_name: str,
    num_workgroups,
    num_waves,
    m,
    n,
    k,
    pass_pipeline,
    mcpu,
    wavefront_size=64,
):
    """Test MFMA end-to-end kernel execution from parsed MLIR file.

    Tests block matrix multiplication where:
    - m, n, k are the number of blocks in each dimension
    - Each block is M_BLOCK_SIZE x N_BLOCK_SIZE x K_BLOCK_SIZE (16x16x16)
    - Overall matrix dimensions: A is (m*M_BLOCK_SIZE) x (k*K_BLOCK_SIZE),
      B is (k*K_BLOCK_SIZE) x (n*N_BLOCK_SIZE), C is (m*M_BLOCK_SIZE) x (n*N_BLOCK_SIZE)
    - num_workgroups: number of workgroups to launch
    - num_waves: number of waves per workgroup
    - Each workgroup/wave needs its own data (batch = num_workgroups * num_waves)
    """

    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(test_dir, "..", mlir_filename)
    library_path = os.path.join(test_dir, "..", "library", "common", "indexing.mlir")

    with ir.Context() as ctx:
        dt_a, dt_b, dt_c = np.float16, np.float16, np.float32
        size_a = np.dtype(dt_a).itemsize  # bytes per element for A
        size_b = np.dtype(dt_b).itemsize  # bytes per element for B

        def preprocess(x: str) -> str:
            x = x.replace("{{SIZE_M}}", str(m))
            x = x.replace("{{SIZE_N}}", str(n))
            x = x.replace("{{SIZE_K}}", str(k))
            x = x.replace(
                "{{LDS_B_SHIFT}}", str((m * k * M_BLOCK_SIZE * K_BLOCK_SIZE * size_a))
            )
            x = x.replace(
                "{{LDS_SIZE}}",
                str(
                    (
                        m * k * M_BLOCK_SIZE * K_BLOCK_SIZE * size_a
                        + k * n * K_BLOCK_SIZE * N_BLOCK_SIZE * size_b
                    )
                ),
            )
            return x

        asm_complete, module_after_passes = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            preprocess=preprocess,
            print_ir_after_all=False,
            library_paths=[library_path],
        )

        # Create matrices with block-major layout: blocks stored contiguously
        # batch by num_workgroups * num_waves since each workgroup/wave needs its own data
        batch = num_workgroups * num_waves
        # A: batch * (m * k) blocks, each block is M_BLOCK_SIZE * K_BLOCK_SIZE contiguous elements
        # B: batch * (k * n) blocks, each block is K_BLOCK_SIZE * N_BLOCK_SIZE contiguous elements
        # C: batch * (m * n) blocks, each block is M_BLOCK_SIZE * N_BLOCK_SIZE contiguous elements
        a_size = batch * (m * k) * (M_BLOCK_SIZE * K_BLOCK_SIZE)
        b_size = batch * (k * n) * (K_BLOCK_SIZE * N_BLOCK_SIZE)
        c_size = batch * (m * n) * (M_BLOCK_SIZE * N_BLOCK_SIZE)
        a_data = np.full(a_size, 1.0, dtype=dt_a)
        b_data = np.full(b_size, 2.0, dtype=dt_b)
        c_data = np.zeros(c_size, dtype=dt_c)

        def verify_fn(input_args, output_args):
            # Convert from block-major to element-major layout for verification
            a_flat = np.array(input_args[0])
            a_blocks = a_flat.reshape(batch, m, k, M_BLOCK_SIZE, K_BLOCK_SIZE)

            b_flat = np.array(input_args[1])
            b_blocks = b_flat.reshape(batch, k, n, K_BLOCK_SIZE, N_BLOCK_SIZE)

            c_flat = np.array(output_args[0])
            c_blocks = c_flat.reshape(batch, m, n, M_BLOCK_SIZE, N_BLOCK_SIZE)

            # Compute reference using block matrix multiplication
            ref = np.zeros((batch, m, n, M_BLOCK_SIZE, N_BLOCK_SIZE), dtype=dt_c)
            for b in range(batch):
                for i in range(m):
                    for j in range(n):
                        for l in range(k):
                            a_block = a_blocks[b, i, l]
                            b_block = b_blocks[b, l, j]
                            ref[b, i, j] = ref[b, i, j] + np.matmul(
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

            num_threads = num_waves * wavefront_size
            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=[a_data, b_data],
                output_args=[c_data],
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                grid_dim=(num_workgroups, 1, 1),
                block_dim=(num_threads, 1, 1),
                verify_fn=verify_fn,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MFMA end-to-end kernel execution with block matrix multiplication"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=4,
        help="Number of blocks in M dimension (default: 4)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of blocks in N dimension (default: 4)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of blocks in K dimension (default: 4)",
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
        default="mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir",
        help="MLIR filename to test (default: mfma_dword4_mxnxk_16x16x16_f16f16f32.mlir)",
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
        kernel_name="test_matmul_kernel",
        m=args.m,
        n=args.n,
        k=args.k,
        num_workgroups=args.num_workgroups,
        num_waves=args.num_waves,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=args.mcpu,
    )
