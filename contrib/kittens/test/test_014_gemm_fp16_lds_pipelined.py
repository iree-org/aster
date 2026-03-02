"""Test: Pipelined LDS GEMM (2/3-stage) via aster-scf-pipeline."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file, pipelined_substitutions


class TestKittensGEMMLDSPipelined:
    """Test GEMM via aster-scf-pipeline (single source -> 2/3-stage pipeline)."""

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_lds_pipelined(self, k, num_stages):
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_014_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions(k, num_stages),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
