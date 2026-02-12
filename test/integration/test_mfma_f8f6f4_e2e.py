"""Integration tests for scaled MFMA (f8f6f4) end-to-end on CDNA4.

Two kernels test distinct failure modes:

Kernel 1 (scaled_mfma_nonidentity_scales):
  A = FP8 E4M3 1.75, B = FP8 E4M3 1.25, C = 0
  scale_A = 4.0 (E8M0 exp 129), scale_B = 2.0 (E8M0 exp 128)
  Expected: 4.0 * 2.0 * 128 * 1.75 * 1.25 = 2240.0
  Tests: scale factors are applied (identity scales would give 280.0)

Kernel 2 (scaled_mfma_split_k_accum):
  A[k=0..63] = 1.0, A[k=64..127] = 1.5
  B[k=0..63] = 2.0, B[k=64..127] = 0.5
  C = 10.0 (f32), scales = identity
  Expected: 64*1.0*2.0 + 64*1.5*0.5 + 10.0 = 186.0
  Tests: full k-range computed, C accumulation works
"""

import os

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx950"
WAVEFRONT_SIZE = 64
MLIR_FILE = "mfma-f8f6f4-e2e.mlir"


class TestScaledMfmaF8f6f4:
    """Test scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4) end-to-end on gfx950."""

    def test_nonidentity_scales(self):
        """A=1.75, B=1.25, scale_A=4.0, scale_B=2.0 -> D = 2240.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 2240.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"Scaled MFMA with non-identity scales failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "scaled_mfma_nonidentity_scales",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
            skip_on_cross_compile=True,
        )

    def test_split_k_with_accumulator(self):
        """Split A/B across k-range + C=10.0 -> D = 186.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 186.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"Scaled MFMA with split k-range + accumulator failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "scaled_mfma_split_k_accum",
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
