"""Test: Single MFMA matmul computes D = A @ B^T correctly."""

import numpy as np

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensMFMA:
    """Test @mfma_f32_16x16x16_f16 function from kittens/global_16x16_f16.mlir."""

    def test_mfma_matmul(self):
        """MFMA should compute D = A @ B^T correctly."""
        A = np.eye(16, dtype=np.float16)
        B = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) / 256.0
        D_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_002_mfma.mlir"),
            kernel_name="test_mfma",
            input_args=[A.flatten(), B.flatten()],
            output_args=[D_output],
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(D_output, expected, rtol=1e-3, atol=1e-3)
