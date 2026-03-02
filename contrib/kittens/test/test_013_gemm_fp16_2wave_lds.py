"""Test: 2-wave GEMM with LDS (XOR swizzle)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensGEMM2WaveLDS:
    """Test 2-wave GEMM with LDS: C[32x16] = A[32xK] @ B[16xK]^T.

    2x1 wave grid with LDS (XOR swizzle):
      - Each wave loads its own A tile into per-wave LDS buffer
      - Both waves redundantly load shared B tile
      - s_barrier synchronizes before LDS reads
    """

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_2wave_lds(self, k):
        """2-wave LDS GEMM should compute C = A @ B^T correctly."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_013_gemm_fp16_2wave_lds.mlir"),
            kernel_name="gemm_2wave_lds",
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
