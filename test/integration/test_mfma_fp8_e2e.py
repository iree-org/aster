"""Integration tests for CDNA3 FP8 MFMA (v_mfma_f32_16x16x32_fp8_fp8) on gfx942.

NOTE: CDNA3 uses FP8 E4M3FNUZ format (bias=8), NOT OCP E4M3 (bias=7).
Value = 2^(E-8) * (1 + M/8) for E>0, or 2^(-7) * (M/8) for E=0.

Two kernels test distinct failure modes:

Kernel 1 (mfma_fp8_ones):
  A = FP8 E4M3FNUZ 1.0 (0x40), B = FP8 E4M3FNUZ 1.0 (0x40), C = 0
  Expected: sum_{k=0}^{31} 1.0 * 1.0 = 32.0
  Tests: basic FP8 MFMA produces correct dot product

Kernel 2 (mfma_fp8_with_accum):
  A = FP8 E4M3FNUZ 1.5 (0x44), B = FP8 E4M3FNUZ 2.0 (0x48), C = 10.0 (f32)
  Expected: sum_{k=0}^{31} 1.5 * 2.0 + 10.0 = 96.0 + 10.0 = 106.0
  Tests: non-trivial FP8 values + f32 accumulation
"""

import os

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
MLIR_FILE = "mfma-fp8-e2e.mlir"


class TestMfmaFp8:
    """Test FP8 MFMA (v_mfma_f32_16x16x32_fp8_fp8) end-to-end on gfx942."""

    def test_ones(self):
        """A=1.0, B=1.0, C=0 -> D = 32.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 32.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"FP8 MFMA with ones failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_fp8_ones",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
            skip_on_cross_compile=True,
        )

    def test_with_accumulator(self):
        """A=1.5, B=2.0, C=10.0 -> D = 106.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 106.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"FP8 MFMA with accumulator failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_fp8_with_accum",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
