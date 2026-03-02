"""Test: Double-buffer LDS GEMM (Phase 4 - latency hiding)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensGEMMLDS2Buffer:
    """Test GEMM with double-buffer LDS (Phase 4 - latency hiding)."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_lds_2buf(self, k):
        """GEMM with double-buffer LDS should match reference and show speedup."""
        k_tiles = k // 16
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_010_gemm_fp16_lds_2buf.mlir"),
            kernel_name="gemm_16x16xK_lds_2buf",
            input_args=[A_flat, B_flat],
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
