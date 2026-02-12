"""Unit tests for kittens/tiles_16x16.mlir library functions."""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from aster.testing import compile_and_run as _compile_and_run
from aster.pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    FUTURE_SROA_PASS_PIPELINE,
)
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


@dataclass
class KittensRunConfig:
    """Runtime config for kernel launches.

    Mutated from __main__ for profiling.
    """

    num_blocks: int = 1
    num_iterations: int = 1


run_config = KittensRunConfig()


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


def compile_and_run(mlir_file, kernel_name, pass_pipeline, input_args, output_args):
    """Compile an MLIR file to HSACO and execute the kernel on GPU."""
    _compile_and_run(
        file_name=mlir_file,
        kernel_name=kernel_name,
        input_data=input_args,
        output_data=output_args,
        pass_pipeline=pass_pipeline,
        library_paths=get_kittens_library_paths(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=(run_config.num_blocks, 1, 1),
        block_dim=(64, 1, 1),
        num_iterations=run_config.num_iterations,
        skip_on_cross_compile=True,
    )


class TestKittensZeroC:
    """Test @zero_C function from kittens/tiles_16x16.mlir."""

    def test_zero_C_produces_zeros(self):
        """Zero-initialized C tile should contain all zeros."""
        output = np.zeros(16 * 16, dtype=np.int32)
        compile_and_run(
            get_mlir_file("test_zero_C.mlir"),
            "test_zero_C",
            TEST_SCF_PIPELINING_PASS_PIPELINE,
            input_args=[],
            output_args=[output],
        )
        expected = np.zeros(16 * 16, dtype=np.int32)
        np.testing.assert_array_equal(output, expected)


class TestKittensLoadStoreA:
    """Test @load_A_f16 and @store_A_f16 functions from kittens/tiles_16x16.mlir."""

    def test_load_store_roundtrip(self):
        """Load A tile and store it back - should preserve original data."""
        input_f16 = np.arange(16 * 16, dtype=np.float16)
        input_data = input_f16.view(np.uint16)
        output_data = np.full(16 * 16, 0xFFFF, dtype=np.uint16)

        compile_and_run(
            get_mlir_file("test_load_store_A.mlir"),
            "test_load_store_A",
            TEST_SCF_PIPELINING_PASS_PIPELINE,
            input_args=[input_data],
            output_args=[output_data],
        )
        np.testing.assert_array_equal(output_data, input_data)


class TestKittensMFMA:
    """Test @mfma_f32_16x16x16_f16 function from kittens/tiles_16x16.mlir."""

    def test_mfma_matmul(self):
        """MFMA should compute D = A @ B^T correctly."""
        A = np.eye(16, dtype=np.float16)
        B = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) / 256.0
        D_output = np.zeros(16 * 16, dtype=np.float32)

        compile_and_run(
            get_mlir_file("test_mfma.mlir"),
            "test_mfma",
            TEST_SCF_PIPELINING_PASS_PIPELINE,
            input_args=[A.flatten(), B.flatten()],
            output_args=[D_output],
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(D_output, expected, rtol=1e-3, atol=1e-3)


def _make_gemm_inputs(K):
    """Create random f16 test matrices for GEMM: A[16xK], B[16xK]."""
    np.random.seed(42)
    A = (np.random.randn(16, K) * 0.05).astype(np.float16)
    B = (np.random.randn(16, K) * 0.05).astype(np.float16)
    return A, B


class TestKittensGEMM:
    """Test GEMM kernel: C[16x16] = A[16x128] @ B[16x128]^T."""

    def test_gemm_16x16x128(self):
        """GEMM should compute C = A @ B^T correctly with K=128."""
        A, B = _make_gemm_inputs(128)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        compile_and_run(
            get_mlir_file("test_gemm_16x16x128.mlir"),
            "gemm_16x16x128",
            TEST_SCF_PIPELINING_PASS_PIPELINE,
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMMSched:
    """Test GEMM with autoschedule + op-scheduling: C[16x16] = A[16x128] @ B[16x128]^T."""

    def test_gemm_16x16x128_sched(self):
        """Scheduled GEMM should produce same result as manually interleaved."""
        A, B = _make_gemm_inputs(128)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        compile_and_run(
            get_mlir_file("test_gemm_16x16x128_with_sched.mlir"),
            "gemm_16x16x128_sched",
            FUTURE_SROA_PASS_PIPELINE,
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run kittens tests")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of workgroups (default: 1, use 304 for full MI300 occupancy)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of kernel launches per test (default: 1)",
    )
    cli_args = parser.parse_args()

    # Override module-level config for all compile_and_run calls
    run_config.num_blocks = cli_args.num_blocks
    run_config.num_iterations = cli_args.num_iterations

    def run_test(test_fn, *args, **kwargs):
        """Run a test, handling pytest.skip gracefully when running without pytest."""
        try:
            test_fn(*args, **kwargs)
        except pytest.skip.Exception as e:
            print(f"  SKIPPED: {e}")

    run_test(TestKittensZeroC().test_zero_C_produces_zeros)
    run_test(TestKittensLoadStoreA().test_load_store_roundtrip)
    run_test(TestKittensMFMA().test_mfma_matmul)
    run_test(TestKittensGEMM().test_gemm_16x16x128)
    run_test(TestKittensGEMMSched().test_gemm_16x16x128_sched)
