"""Integration tests for AGPR-backed MFMA (v_mfma_f32_16x16x16_f16) on gfx942.

Tests that AGPR init via v_accvgpr_write, MFMA with AGPR accumulators,
and direct AGPR-to-global store produce correct numerical results.

Kernel 1 (mfma_agpr_ones):
  A = f16 1.0, B = f16 1.0, C = 0 (in AGPRs)
  Expected: D = sum_{k=0}^{15} 1.0 * 1.0 + 0 = 16.0

Kernel 2 (mfma_agpr_with_accum):
  A = f16 1.0, B = f16 2.0, C = f32 10.0 (in AGPRs)
  Expected: D = sum_{k=0}^{15} 1.0 * 2.0 + 10.0 = 32.0 + 10.0 = 42.0
"""

import os

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
MLIR_FILE = "mfma-agpr-f16-e2e.mlir"


class TestMfmaAgprs:
    """Test AGPR-backed MFMA (v_mfma_f32_16x16x16_f16) end-to-end on gfx942."""

    def test_ones(self):
        """A=f16 1.0, B=f16 1.0, C=0 (AGPRs) -> D = 16.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 16.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"AGPR MFMA with ones failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_agpr_ones",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )

    def test_with_accumulator(self):
        """A=f16 1.0, B=f16 2.0, C=f32 10.0 (AGPRs) -> D = 42.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 42.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"AGPR MFMA with accumulator failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_agpr_with_accum",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
