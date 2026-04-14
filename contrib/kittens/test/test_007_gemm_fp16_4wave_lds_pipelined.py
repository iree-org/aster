"""Test: 4-wave GEMM with pipelined LDS + AGPR accumulators.

Uses compute_16x16_f16.mlir (AGPR) with fire-and-forget stores.
"""

import numpy as np
import pytest

from aster.test_pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    pipelined_substitutions_16x32,
)


class TestKittensGEMM4WaveLDSPipelined_AGPR:
    """Test 4-wave GEMM with pipelined LDS + AGPR: C[32x32] = A[32xK] @ B[32xK]^T.

    2x2 wave grid with XOR-swizzle LDS and sched.stage annotations.
    """

    @pytest.mark.parametrize("pipeline_strategy", [1, 3], ids=["ps1", "ps3"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_4wave_lds_pipelined(self, k, pipeline_strategy, print_ir_after_all=False):
        """4-wave pipelined LDS GEMM with AGPR should compute C = A @ B^T correctly."""
        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mcpu="gfx942",
            mlir_file=get_mlir_file("test_007_gemm_fp16_4wave_lds_pipelined.mlir"),
            kernel_name="gemm_4wave_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(256, 1, 1),
            template_substitutions=pipelined_substitutions_16x32(k, pipeline_strategy),
            library_paths=get_kittens_16x16_lds_library_paths(),
            print_ir_after_all=print_ir_after_all,
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
    TestKittensGEMM4WaveLDSPipelined_AGPR().test_gemm_4wave_lds_pipelined(
        a.k_scaling_factor * 32, a.pipeline_strategy, print_ir_after_all=a.print_ir_after_all
    )
