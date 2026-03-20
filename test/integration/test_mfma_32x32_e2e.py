"""Integration test for v_mfma_f32_32x32x8_f16 on gfx942 (CDNA3).

A = f16 1.0, B = f16 1.0, C = 0
Expected: D[i,j] = sum_{k=0}^{7} 1.0 * 1.0 = 8.0 for all elements.

32x32 MFMA lane mapping (16 VGPRs per lane):
  row = lane_id % 32
  col = (lane_id / 32) * 16 + vgpr_index

Each of 64 lanes stores 16 f32 values = 64 bytes.
Total output: 64 * 16 = 1024 f32 values (32 * 32 matrix).
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
MLIR_FILE = "mfma-32x32-e2e.mlir"


def test_mfma_32x32x8_ones():
    """A=f16 1.0, B=f16 1.0, C=0 -> D = 8.0 for all 32x32 elements."""
    # 64 lanes * 16 dwords each = 1024 f32 values
    total_elements = WAVEFRONT_SIZE * 16
    output = np.zeros(total_elements, dtype=np.float32)

    def verify(inputs, outputs):
        result = np.array(outputs[0])
        expected = 8.0
        np.testing.assert_allclose(
            result,
            expected,
            err_msg=(
                f"32x32x8 MFMA with ones failed: "
                f"expected {expected}, "
                f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
            ),
        )

    compile_and_run(
        MLIR_FILE,
        "mfma_32x32x8_ones",
        output_data=[output],
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        verify_fn=verify,
        library_paths=[_REGISTER_INIT],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
