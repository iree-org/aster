"""Unit tests for kittens/tiles_16x16.mlir library functions."""

import os
import pytest
import numpy as np
from typing import List

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)
from aster.pass_pipelines import TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_kittens_library_paths() -> List[str]:
    """Get paths to all required library files including kittens."""
    base_paths = get_library_paths()
    kittens_dir = os.path.join(os.path.dirname(__file__), "..", "library")
    kittens_paths = [
        os.path.join(kittens_dir, "tiles_16x16.mlir"),
    ]
    return base_paths + kittens_paths


def get_mlir_file(file_name: str) -> str:
    """Get path to a test MLIR file in the kittens test directory."""
    return os.path.join(os.path.dirname(__file__), file_name)


class TestKittensZeroC:
    """Test @zero_C function from kittens/tiles_16x16.mlir."""

    def test_zero_C_produces_zeros(self):
        """Zero-initialized C tile should contain all zeros."""
        mlir_file = get_mlir_file("test_zero_C.mlir")
        kernel_name = "test_zero_C"
        library_paths = get_kittens_library_paths()

        # Output: 16x16 matrix of f32 (as int32 for bit-exact comparison)
        output = np.zeros(16 * 16, dtype=np.int32)

        with ir.Context() as ctx:
            asm_complete, module = compile_mlir_file_to_asm(
                mlir_file,
                kernel_name,
                TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
                ctx,
                library_paths=library_paths,
            )

            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")

            with hsaco_file(hsaco_path):
                if not utils.system_has_mcpu(mcpu=MCPU):
                    print(asm_complete)
                    pytest.skip(f"GPU {MCPU} not available")

                execute_kernel_and_verify(
                    hsaco_path=hsaco_path,
                    kernel_name=kernel_name,
                    input_args=[],
                    output_args=[output],
                    mcpu=MCPU,
                    wavefront_size=WAVEFRONT_SIZE,
                    grid_dim=(1, 1, 1),
                    block_dim=(64, 1, 1),
                )

        # All values should be zero
        expected = np.zeros(16 * 16, dtype=np.int32)
        np.testing.assert_array_equal(output, expected)


class TestKittensLoadStoreA:
    """Test @load_A_f16 and @store_A_f16 functions from kittens/tiles_16x16.mlir."""

    def test_load_store_roundtrip(self):
        """Load A tile and store it back - should preserve original data."""
        mlir_file = get_mlir_file("test_load_store_A.mlir")
        kernel_name = "test_load_store_A"
        library_paths = get_kittens_library_paths()

        # Input: 16x16 matrix of f16 with sequential values (as uint16)
        # Use a recognizable pattern: values 0, 1, 2, ..., 255
        input_data = np.arange(16 * 16, dtype=np.uint16)
        # Convert to f16 bit pattern (just use the raw uint16 values as f16 bits)
        # For testing, we use simple integer values that are valid f16
        input_f16 = np.arange(16 * 16, dtype=np.float16)
        input_data = input_f16.view(np.uint16)

        # Output: same size, initialized to different value
        output_data = np.full(16 * 16, 0xFFFF, dtype=np.uint16)

        with ir.Context() as ctx:
            asm_complete, module = compile_mlir_file_to_asm(
                mlir_file,
                kernel_name,
                TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
                ctx,
                library_paths=library_paths,
            )

            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")

            with hsaco_file(hsaco_path):
                if not utils.system_has_mcpu(mcpu=MCPU):
                    print(asm_complete)
                    pytest.skip(f"GPU {MCPU} not available")

                execute_kernel_and_verify(
                    hsaco_path=hsaco_path,
                    kernel_name=kernel_name,
                    input_args=[input_data],
                    output_args=[output_data],
                    mcpu=MCPU,
                    wavefront_size=WAVEFRONT_SIZE,
                    grid_dim=(1, 1, 1),
                    block_dim=(64, 1, 1),
                )

        # Output should match input exactly (bit-for-bit)
        np.testing.assert_array_equal(output_data, input_data)


class TestKittensMFMA:
    """Test @mfma_f32_16x16x16_f16 function from kittens/tiles_16x16.mlir."""

    def test_mfma_matmul(self):
        """MFMA should compute D = A @ B^T correctly."""
        mlir_file = get_mlir_file("test_mfma.mlir")
        kernel_name = "test_mfma"
        library_paths = get_kittens_library_paths()

        # Create simple test matrices (16x16 f16)
        # A = identity-like pattern, B = sequential values
        A = np.eye(16, dtype=np.float16)  # Identity matrix
        B = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) / 256.0

        # Flatten for kernel input
        A_flat = A.flatten()
        B_flat = B.flatten()

        # Output: 16x16 f32
        D_output = np.zeros(16 * 16, dtype=np.float32)

        with ir.Context() as ctx:
            asm_complete, module = compile_mlir_file_to_asm(
                mlir_file,
                kernel_name,
                TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
                ctx,
                library_paths=library_paths,
            )

            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")

            with hsaco_file(hsaco_path):
                if not utils.system_has_mcpu(mcpu=MCPU):
                    print(asm_complete)
                    pytest.skip(f"GPU {MCPU} not available")

                execute_kernel_and_verify(
                    hsaco_path=hsaco_path,
                    kernel_name=kernel_name,
                    input_args=[A_flat, B_flat],
                    output_args=[D_output],
                    mcpu=MCPU,
                    wavefront_size=WAVEFRONT_SIZE,
                    grid_dim=(1, 1, 1),
                    block_dim=(64, 1, 1),
                )

        # Expected: D = A @ B^T (note: MFMA computes A @ B^T, not A @ B)
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(D_output, expected, rtol=1e-3, atol=1e-3)


class TestKittensGEMM:
    """Test minimal GEMM kernel: C[16x16] = A[16x32] @ B[16x32]^T."""

    def test_gemm_16x16x32(self):
        """GEMM should compute C = A @ B^T correctly with K=32."""
        mlir_file = get_mlir_file("test_gemm_16x16x32.mlir")
        kernel_name = "gemm_16x16x32"
        library_paths = get_kittens_library_paths()

        # Create test matrices
        # A: 16x32 f16, B: 16x32 f16
        # Use small values to avoid overflow in f16
        np.random.seed(42)  # Reproducibility
        A = (np.random.randn(16, 32) * 0.1).astype(np.float16)
        B = (np.random.randn(16, 32) * 0.1).astype(np.float16)

        # Flatten for kernel input
        A_flat = A.flatten()
        B_flat = B.flatten()

        # Output: 16x16 f32
        C_output = np.zeros(16 * 16, dtype=np.float32)

        with ir.Context() as ctx:
            asm_complete, module = compile_mlir_file_to_asm(
                mlir_file,
                kernel_name,
                TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
                ctx,
                library_paths=library_paths,
            )

            hsaco_path = utils.assemble_to_hsaco(
                asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
            )
            if hsaco_path is None:
                raise RuntimeError("Failed to assemble kernel to HSACO")

            with hsaco_file(hsaco_path):
                if not utils.system_has_mcpu(mcpu=MCPU):
                    print(asm_complete)
                    pytest.skip(f"GPU {MCPU} not available")

                execute_kernel_and_verify(
                    hsaco_path=hsaco_path,
                    kernel_name=kernel_name,
                    input_args=[A_flat, B_flat],
                    output_args=[C_output],
                    mcpu=MCPU,
                    wavefront_size=WAVEFRONT_SIZE,
                    grid_dim=(1, 1, 1),
                    block_dim=(64, 1, 1),
                )

        # Expected: C = A @ B^T (MFMA computes A @ B^T)
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    TestKittensZeroC().test_zero_C_produces_zeros()
    TestKittensLoadStoreA().test_load_store_roundtrip()
    TestKittensMFMA().test_mfma_matmul()
    TestKittensGEMM().test_gemm_16x16x32()
