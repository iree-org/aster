"""Test: Fixed-K GEMM (K=128) computes C = A @ B^T."""

import numpy as np

from kittens_helpers import run_kittens_kernel, get_mlir_file, _make_gemm_inputs


class TestKittensGEMM:
    """Test GEMM kernel: C[16x16] = A[16x128] @ B[16x128]^T."""

    def test_gemm_16x16x128(self):
        """GEMM should compute C = A @ B^T correctly with K=128."""
        A, B = _make_gemm_inputs(128)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_003_gemm_fp16_fixed.mlir"),
            kernel_name="gemm_16x16x128",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
