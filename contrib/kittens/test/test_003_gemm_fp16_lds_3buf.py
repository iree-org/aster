"""Test: Triple-buffer LDS GEMM with AGPR accumulators (lds_16x32 tiles).

Uses lds_16x32_f16.mlir: dwordx4 global loads, XOR-swizzled LDS.
Each 16x32 tile covers K=32, yielding 2 MFMA K-steps per iteration.
"""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
)


class TestKittensGEMMLDS3Buffer_AGPR:
    """Test GEMM with triple-buffer LDS (16x32 tiles) + AGPR accumulators."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_gemm_lds_3buf(self, k, print_ir_after_all=False):
        """GEMM with triple-buffer LDS + AGPR should match reference."""
        k_tiles = k // 32
        stride_ab = k * 2

        np.random.seed(42 + k)
        A = (np.random.randn(16, k) * 0.1).astype(np.float16)
        B = (np.random.randn(16, k) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_003_gemm_fp16_lds_3buf.mlir"),
            kernel_name="gemm_16x16xK_lds_3buf",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
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
    TestKittensGEMMLDS3Buffer_AGPR().test_gemm_lds_3buf(
        a.k_scaling_factor * 32, print_ir_after_all=a.print_ir_after_all
    )
