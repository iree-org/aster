"""Test: Pipelined LDS GEMM with 32x32x8 MFMA and 32x32 transfer tiles."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_32x32_library_paths,
    pipelined_substitutions_32x32,
)


class TestGEMM32x32LDSPipelined:
    """Test 32x32 GEMM via aster-scf-pipeline with 32x32 transfer tiles."""

    @pytest.mark.parametrize(
        "num_stages", [2, 3, 4], ids=["2stage", "3stage", "4stage"]
    )
    @pytest.mark.parametrize("k", [64, 128, 256])
    def test_gemm_32x32_lds_pipelined(self, k, num_stages):
        k_tiles = k // 32
        if k_tiles < num_stages:
            pytest.skip(
                f"K={k} gives {k_tiles} tiles, need >= {num_stages} for {num_stages}-stage pipeline"
            )
        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(32, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_026_gemm_32x32_lds_pipelined.mlir"),
            kernel_name="gemm_32x32xK_lds_pipelined",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
            template_substitutions=pipelined_substitutions_32x32(
                k, num_stages, k_per_tile=32
            ),
            library_paths=get_kittens_32x32_library_paths(),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
