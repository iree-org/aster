"""Test: 2x2 multi-tile GEMM with pipelined LDS + AGPR accumulators.

Mirrors test_006_gemm_fp16_multitile_lds_pipelined.py but uses
compute_16x16_f16.mlir (AGPR) with fire-and-forget stores.
"""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    pipelined_substitutions_16x32,
)


class TestKittensGEMMMultiTileLDSPipelined_AGPR:
    """Test single-wave 2x2 multi-tile GEMM with pipelined LDS + AGPR:
    C[32x32] = A[32xK] @ B[32xK]^T.

    One wave computes a 2x2 grid of 16x16 MFMA tiles using LDS staging with XOR swizzle
    and sched.stage annotations for automatic pipelining. 4 LDS tiles per stage (2 A + 2
    B), 4 MFMAs per K iteration. AGPR accumulators with fire-and-forget stores.
    """

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("k", [96, 128])
    def test_gemm_multitile_lds_pipelined(
        self, k, num_stages, print_ir_after_all=False
    ):
        """Single-wave 2x2 multi-tile LDS pipelined GEMM with AGPR should match reference."""
        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_006_gemm_fp16_multitile_lds_pipelined.mlir"),
            kernel_name="gemm_multitile_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_16x32(k, num_stages),
            library_paths=get_kittens_16x16_lds_library_paths(),
            print_ir_after_all=print_ir_after_all,
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
    TestKittensGEMMMultiTileLDSPipelined_AGPR().test_gemm_multitile_lds_pipelined(
        a.k_scaling_factor * 32, a.num_stages, print_ir_after_all=a.print_ir_after_all
    )
