"""Unit tests for kittens/tiles_16x16.mlir library functions."""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict

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


def run_kittens_kernel(
    mlir_file,
    kernel_name,
    input_args=None,
    output_args=None,
    pass_pipeline=None,
    template_substitutions=None,
    block_dim=(64, 1, 1),
):
    """Compile an MLIR file to HSACO and execute the kernel on GPU."""
    preprocess = None
    if template_substitutions:
        subs = template_substitutions

        def preprocess(content):
            for pattern, replacement in subs.items():
                content = content.replace(pattern, replacement)
            return content

    _compile_and_run(
        file_name=mlir_file,
        kernel_name=kernel_name,
        input_data=input_args,
        output_data=output_args,
        pass_pipeline=pass_pipeline,
        preprocess=preprocess,
        library_paths=get_kittens_library_paths(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=(run_config.num_blocks, 1, 1),
        block_dim=block_dim,
        num_iterations=run_config.num_iterations,
        skip_on_cross_compile=True,
        print_ir_after_all=True,
    )


class TestKittensZeroC:
    """Test @zero_C function from kittens/tiles_16x16.mlir."""

    def test_zero_C_produces_zeros(self):
        """Zero-initialized C tile should contain all zeros."""
        output = np.zeros(16 * 16, dtype=np.int32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_zero_C.mlir"),
            kernel_name="test_zero_C",
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

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_load_store_A.mlir"),
            kernel_name="test_load_store_A",
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

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_mfma.mlir"),
            kernel_name="test_mfma",
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

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_16x16x128.mlir"),
            kernel_name="gemm_16x16x128",
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

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_16x16x128_with_sched.mlir"),
            kernel_name="gemm_16x16x128_sched",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=FUTURE_SROA_PASS_PIPELINE,
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMMLoop:
    """Test GEMM with scf.for K-loop for arbitrary K values."""

    @pytest.mark.parametrize("k", [128, 4096])
    def test_gemm_16x16xK(self, k):
        """GEMM should compute C = A @ B^T for various K values."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_16x16xK.mlir"),
            kernel_name="gemm_16x16xK",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions={
                "{{K}}": str(k),
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMM2Wave:
    """Test 2-wave GEMM: C[32x16] = A[32xK] @ B[16xK]^T.

    2x1 wave grid: wave 0 computes C[0:16, 0:16], wave 1 computes C[16:32, 0:16].
    Both waves share the same B matrix; each loads its own A rows.
    """

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_2wave(self, k):
        """2-wave GEMM should compute C = A @ B^T correctly."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_2wave.mlir"),
            kernel_name="gemm_2wave",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(128, 1, 1),
            template_substitutions={
                "{{K}}": str(k),
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMM4Wave:
    """Test 4-wave GEMM: C[32x32] = A[32xK] @ B[32xK]^T.

    2x2 wave grid:
      Wave 0: C[0:16, 0:16],   Wave 1: C[0:16, 16:32]
      Wave 2: C[16:32, 0:16],  Wave 3: C[16:32, 16:32]
    Each wave loads its own A rows and B rows based on grid position.
    """

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_4wave(self, k):
        """4-wave GEMM should compute C = A @ B^T correctly."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_4wave.mlir"),
            kernel_name="gemm_4wave",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(256, 1, 1),
            template_substitutions={
                "{{K}}": str(k),
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    # Registry: (kernel_name, test_fn, args, kwargs)
    ALL_TESTS = [
        ("test_zero_C", TestKittensZeroC().test_zero_C_produces_zeros, [], {}),
        (
            "test_load_store_A",
            TestKittensLoadStoreA().test_load_store_roundtrip,
            [],
            {},
        ),
        ("test_mfma", TestKittensMFMA().test_mfma_matmul, [], {}),
        ("gemm_16x16x128", TestKittensGEMM().test_gemm_16x16x128, [], {}),
        (
            "gemm_16x16x128_sched",
            TestKittensGEMMSched().test_gemm_16x16x128_sched,
            [],
            {},
        ),
        ("gemm_16x16xK_k128", TestKittensGEMMLoop().test_gemm_16x16xK, [], {"k": 128}),
        (
            "gemm_16x16xK_k4096",
            TestKittensGEMMLoop().test_gemm_16x16xK,
            [],
            {"k": 4096},
        ),
        ("gemm_2wave_k32", TestKittensGEMM2Wave().test_gemm_2wave, [], {"k": 32}),
        ("gemm_2wave_k64", TestKittensGEMM2Wave().test_gemm_2wave, [], {"k": 64}),
        ("gemm_2wave_k128", TestKittensGEMM2Wave().test_gemm_2wave, [], {"k": 128}),
        ("gemm_4wave_k32", TestKittensGEMM4Wave().test_gemm_4wave, [], {"k": 32}),
        ("gemm_4wave_k64", TestKittensGEMM4Wave().test_gemm_4wave, [], {"k": 64}),
        ("gemm_4wave_k128", TestKittensGEMM4Wave().test_gemm_4wave, [], {"k": 128}),
    ]

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
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run only tests whose name contains this substring (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test names and exit",
    )
    cli_args = parser.parse_args()

    # Override module-level config for all compile_and_run calls
    run_config.num_blocks = cli_args.num_blocks
    run_config.num_iterations = cli_args.num_iterations

    tests = ALL_TESTS
    if cli_args.test:
        tests = [(n, f, a, kw) for n, f, a, kw in ALL_TESTS if cli_args.test in n]
        if not tests:
            print(f"No tests matching '{cli_args.test}'. Available:")
            for name, _, _, _ in ALL_TESTS:
                print(f"  {name}")
            raise SystemExit(1)

    for name, test_fn, args, kwargs in tests:
        try:
            test_fn(*args, **kwargs)
        except pytest.skip.Exception as e:
            print(f"  SKIPPED: {e}")
