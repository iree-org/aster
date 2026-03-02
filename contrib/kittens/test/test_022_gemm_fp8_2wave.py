"""Test: 2-wave FP8 GEMM."""

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


class TestKittensGEMMFP8_2Wave:
    """Test 2-wave FP8 GEMM: C[32x16] = A[32xK] @ B[16xK]^T.

    2x1 wave grid: wave 0 computes C[0:16, 0:16], wave 1 computes C[16:32, 0:16].
    Both waves share the same B matrix; each loads its own A rows.
    """

    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_fp8_2wave(self, k):
        """2-wave FP8 GEMM should compute C = A @ B^T correctly."""
        A_fp8 = _make_fp8_inputs(32, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(16, k, seed=137 + k)
        C_output = np.zeros(32 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_022_gemm_fp8_2wave.mlir"),
            kernel_name="gemm_fp8_2wave",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(128, 1, 1),
            template_substitutions=_fp8_template_subs(k),
        )

        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(32, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(16, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
