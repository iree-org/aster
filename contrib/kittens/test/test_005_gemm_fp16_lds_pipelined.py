"""Test: Pipelined LDS GEMM (2/3-stage) via aster-scf-pipeline + AGPR accumulators.

Uses lds_16x32_f16.mlir: 4-stage pipeline (GLOBAL_LOAD, DS_WRITE, DS_READ, COMPUTE).
"""

import numpy as np
import pytest

from aster.test_pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    TEST_SCF_PIPELINING_LL_SCHED_PASS_PIPELINE,
)

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
    def test_gemm_lds_pipelined(self, k, num_stages, print_ir_after_all=False):
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, num_stages),
            library_paths=get_kittens_16x16_lds_library_paths(),
            print_ir_after_all=print_ir_after_all,
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_lds_pipelined_ll_sched(self, k, num_stages):
        """Same as test_gemm_lds_pipelined but with ll-sched enabled."""
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_LL_SCHED_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, num_stages),
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k-scaling-factor", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--print-ir-after-all", action="store_true")
    a = parser.parse_args()
    TestKittensGEMMLDSPipelined_AGPR().test_gemm_lds_pipelined(
        a.k_scaling_factor * 32, a.num_stages, print_ir_after_all=a.print_ir_after_all
    )
