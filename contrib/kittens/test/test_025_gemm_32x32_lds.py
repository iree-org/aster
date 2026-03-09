"""Test: LDS GEMM with 32x32x8 MFMA and 32x32 transfer tiles."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_32x32_library_paths,
)


class TestGEMM32x32LDS:
    """Test GEMM with 32x32 transfer tiles (4 MFMAs per K-tile)."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_32x32_lds(self, k):
        """32x32 GEMM with 32x32 transfer tiles should match reference."""
        k_tiles = k // 32
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_025_gemm_32x32_lds.mlir"),
            kernel_name="gemm_32x32xK_lds",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions={
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
            library_paths=get_kittens_32x32_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
