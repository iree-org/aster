"""Test: Pipelined LDS GEMM (2/3-stage) via aster-scf-pipeline + AGPR accumulators.

Uses lds_16x32_f16.mlir: 4-stage pipeline (GLOBAL_LOAD, DS_WRITE, DS_READ, COMPUTE).
"""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    pipelined_substitutions_16x32,
    get_kittens_16x16_lds_library_paths,
)


class TestKittensGEMMLDSPipelined_AGPR:
    """Test GEMM via aster-scf-pipeline with AGPR accumulators + lds_16x32 tiles."""

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("k", [96, 128])
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
            template_substitutions=pipelined_substitutions_16x32(k, num_stages),
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
