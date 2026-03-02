"""Test: FP8 GEMM with single-buffer LDS."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    fp8_e4m3fnuz_to_float,
    _fp8_template_subs,
    _make_fp8_inputs,
)


class TestKittensGEMMFP8_LDS1Buf:
    """Test FP8 GEMM with single-buffer LDS: C = A @ B^T.

    Global -> LDS -> Register -> MFMA pipeline.
    Baseline LDS implementation to establish correctness (no latency hiding).
    """

    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_fp8_lds_1buf(self, k):
        """FP8 LDS GEMM should match reference."""
        A_fp8 = _make_fp8_inputs(16, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(16, k, seed=137 + k)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_024_gemm_fp8_lds_1buf.mlir"),
            kernel_name="gemm_fp8_16x16xK_lds_1buf",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=_fp8_template_subs(k),
        )

        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(16, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(16, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
