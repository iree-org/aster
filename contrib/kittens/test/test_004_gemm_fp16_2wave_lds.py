"""Test: 2-wave GEMM with LDS (XOR swizzle) + AGPR accumulators."""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
)


class TestKittensGEMM2WaveLDS_AGPR:
    """Test 2-wave GEMM with LDS + AGPR: C[32x16] = A[32xK] @ B[16xK]^T.

    Mirrors TestKittensGEMM2WaveLDS but uses AGPR accumulators and
    fire-and-forget store (no write_token).

    2x1 wave grid with LDS (XOR swizzle):
      - Each wave loads its own A tile into per-wave LDS buffer
      - Both waves redundantly load shared B tile
      - s_barrier synchronizes before LDS reads
    """

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_2wave_lds(self, k, print_ir_after_all=False):
        """2-wave LDS GEMM with AGPR should compute C = A @ B^T correctly."""
        k_tiles = k // 32
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(32, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(32 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_004_gemm_fp16_2wave_lds.mlir"),
            kernel_name="gemm_2wave_lds",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(128, 1, 1),
            template_substitutions={
                "{{K}}": str(k),
                "{{K_TILES}}": str(k_tiles),
                "{{STRIDE_AB}}": str(stride_ab),
            },
            library_paths=get_kittens_16x16_lds_library_paths(),
            print_ir_after_all=print_ir_after_all,
        )

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k-scaling-factor", type=int, default=4)
    parser.add_argument("--print-ir-after-all", action="store_true")
    a = parser.parse_args()
    TestKittensGEMM2WaveLDS_AGPR().test_gemm_2wave_lds(
        a.k_scaling_factor * 32, print_ir_after_all=a.print_ir_after_all
    )
