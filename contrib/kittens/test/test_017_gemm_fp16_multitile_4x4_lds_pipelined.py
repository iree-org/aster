"""Test: 4x4 multi-tile GEMM with pipelined LDS (register pressure test)."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file, pipelined_substitutions


class TestKittensGEMMMultiTile4x4LDSPipelined:
    """Test single-wave 4x4 multi-tile GEMM with pipelined LDS:
    C[64x64] = A[64xK] @ B[64xK]^T.

    One wave computes a 4x4 grid of 16x16 MFMA tiles using LDS staging.
    8 LDS tiles per stage (4 A + 4 B), 16 MFMAs per K iteration.
    Tests register pressure limits: 64 VGPRs for accumulators alone.
    """

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_multitile_4x4_lds_pipelined(self, k, num_stages):
        """Single-wave 4x4 multi-tile LDS pipelined GEMM should match reference."""
        np.random.seed(42 + k)
        A = (np.random.randn(64, k) * 0.1).astype(np.float16)
        B = (np.random.randn(64, k) * 0.1).astype(np.float16)
        C_output = np.zeros(64 * 64, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file(
                "test_017_gemm_fp16_multitile_4x4_lds_pipelined.mlir"
            ),
            kernel_name="gemm_multitile_4x4_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions(k, num_stages),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
