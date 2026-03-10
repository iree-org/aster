"""Test: Double-buffer LDS GEMM with AGPR accumulators (lds_16x32 tiles).

Uses lds_16x32_f16.mlir: dwordx4 global loads, XOR-swizzled LDS.
Each 16x32 tile covers K=32, yielding 2 MFMA K-steps per iteration.
"""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
)


class TestKittensGEMMLDS2Buffer_AGPR:
    """Test GEMM with double-buffer LDS (16x32 tiles) + AGPR accumulators."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_lds_2buf(self, k):
        """GEMM with double-buffer LDS + AGPR should match reference."""
        k_tiles = k // 32
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_010_gemm_fp16_lds_2buf.mlir"),
            kernel_name="gemm_16x16xK_lds_2buf",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions={
                "{{K}}": str(k),
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
