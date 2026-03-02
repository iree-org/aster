"""Test: Constexpr multi-tile GEMM with LDS pipelining."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

from kittens_helpers import run_kittens_kernel, get_mlir_file, constexpr_substitutions


class TestKittensGEMMConstexpr:
    """Test constexpr multi-tile GEMM with LDS pipelining.

    Single MLIR template (test_gemm_constexpr.mlir) with scalar-only substitutions for
    arbitrary M_T x N_T tile grids (1x1 through 4x4+). The compiler pipeline (constexpr-
    expansion -> sroa -> mem2reg -> promote-loop-carried-memrefs) eliminates all
    structural complexity.
    """

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize(
        "m_tiles,n_tiles",
        [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (4, 4)],
        ids=["1x1", "1x2", "2x1", "2x2", "3x3", "4x4"],
    )
    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_constexpr(self, k, m_tiles, n_tiles, num_stages):
        """Constexpr GEMM should compute C = A @ B^T correctly."""
        m_dim = m_tiles * 16
        n_dim = n_tiles * 16

        np.random.seed(42 + k + m_tiles * 100 + n_tiles * 10)
        A = (np.random.randn(m_dim, k) * 0.1).astype(np.float16)
        B = (np.random.randn(n_dim, k) * 0.1).astype(np.float16)
        C_output = np.zeros(m_dim * n_dim, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_019_gemm_fp16_constexpr.mlir"),
            kernel_name="gemm_constexpr",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
            template_substitutions=constexpr_substitutions(
                m_tiles, n_tiles, k, num_stages
            ),
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
