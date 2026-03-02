"""Test: GEMM with autoschedule + op-scheduling."""

import numpy as np

from aster.pass_pipelines import FUTURE_SROA_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file, _make_gemm_inputs


class TestKittensGEMMSched:
    """Test GEMM with autoschedule + op-scheduling: C[16x16] = A[16x128] @ B[16x128]^T."""

    def test_gemm_16x16x128_sched(self):
        """Scheduled GEMM should produce same result as manually interleaved."""
        A, B = _make_gemm_inputs(128)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_004_gemm_fp16_sched.mlir"),
            kernel_name="gemm_16x16x128_sched",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=FUTURE_SROA_PASS_PIPELINE,
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
