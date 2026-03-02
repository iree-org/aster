"""Test: 2-wave GEMM (no LDS)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file


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
            mlir_file=get_mlir_file("test_006_gemm_fp16_2wave.mlir"),
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
