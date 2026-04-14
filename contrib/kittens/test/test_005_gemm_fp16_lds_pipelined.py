"""Test: Pipelined LDS GEMM via aster-scf-pipeline + AGPR accumulators.

Uses lds_16x32_f16.mlir with pipeline strategy from PIPELINE_STRATEGIES dict.
"""

import numpy as np
import pytest

from aster.test_pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    TEST_SCF_PIPELINING_LL_SCHED_PASS_PIPELINE,
    TEST_SCF_PIPELINING_HOIST_WAIT_PASS_PIPELINE,
    TEST_SCF_PIPELINING_LL_SCHED_HOIST_WAIT_PASS_PIPELINE,
)

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    pipelined_substitutions_16x32,
    get_kittens_16x16_lds_library_paths,
)


class TestKittensGEMMLDSPipelined_AGPR:
    """Test GEMM via aster-scf-pipeline with AGPR accumulators + lds_16x32 tiles."""

    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_lds_pipelined(self, k, pipeline_strategy, print_ir_after_all=False):
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mcpu="gfx942",
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, pipeline_strategy),
            library_paths=get_kittens_16x16_lds_library_paths(),
            print_ir_after_all=print_ir_after_all,
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_lds_pipelined_ll_sched(self, k, pipeline_strategy):
        """Same as test_gemm_lds_pipelined but with ll-sched enabled."""
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mcpu="gfx942",
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_LL_SCHED_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, pipeline_strategy),
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_lds_pipelined_hoist_wait(self, k, pipeline_strategy):
        """Same as test_gemm_lds_pipelined but with hoist-iter-arg-waits enabled."""
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mcpu="gfx942",
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_HOIST_WAIT_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, pipeline_strategy),
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_lds_pipelined_ll_sched_hoist_wait(self, k, pipeline_strategy):
        """Both ll-sched and hoist-iter-arg-waits enabled."""
        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mcpu="gfx942",
            mlir_file=get_mlir_file("test_005_gemm_fp16_lds_pipelined.mlir"),
            kernel_name="gemm_16x16xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_LL_SCHED_HOIST_WAIT_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, pipeline_strategy),
            library_paths=get_kittens_16x16_lds_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k-scaling-factor", type=int, default=4)
    parser.add_argument("--pipeline-strategy", type=int, default=1)
    parser.add_argument("--print-ir-after-all", action="store_true")
    a = parser.parse_args()
    TestKittensGEMMLDSPipelined_AGPR().test_gemm_lds_pipelined(
        a.k_scaling_factor * 32, a.pipeline_strategy, print_ir_after_all=a.print_ir_after_all
    )
