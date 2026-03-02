"""Test: Single-wave 2x2 multi-tile GEMM (no LDS)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensGEMMMultiTileDirect:
    """Test single-wave multi-tile GEMM: C[32x32] = A[32xK] @ B[32xK]^T.

    One wave computes a 2x2 grid of 16x16 MFMA tiles using 4 iter_args.
    Per K iteration: 4 loads (2 A + 2 B), 4 MFMAs.
    A/B reuse: A[i] reused across N, B[j] reused across M.
    """

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_multitile_direct(self, k):
        """Single-wave 2x2 multi-tile GEMM should match reference."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_015_gemm_fp16_multitile_direct.mlir"),
            kernel_name="gemm_multitile_direct",
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
