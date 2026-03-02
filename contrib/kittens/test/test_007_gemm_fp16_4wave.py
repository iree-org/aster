"""Test: 4-wave GEMM (no LDS)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file


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
            mlir_file=get_mlir_file("test_007_gemm_fp16_4wave.mlir"),
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
