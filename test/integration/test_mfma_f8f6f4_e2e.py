"""Integration test for f8f6f4 MFMA end-to-end kernel execution on CDNA4.

The kernel fills A with FP8 E4M3 1.0, B with FP8 E4M3 2.0, C with 0,
runs v_mfma_f32_16x16x128_f8f6f4, and stores the 16x16 F32 result.
Expected: D[m][n] = sum_k(1.0 * 2.0) for k=0..127 = 256.0 for all elements.
"""

import os

import numpy as np

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx950"
WAVEFRONT_SIZE = 64
KERNEL_NAME = "mfma_f8f6f4_kernel"


class TestMfmaF8f6f4:
    """Test v_mfma_f32_16x16x128_f8f6f4 end-to-end on gfx950."""

    def test_mfma_f8f6f4_all_256(self):
        """A=1.0(FP8), B=2.0(FP8), C=0 -> D = 256.0 for all elements."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 256.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"MFMA f8f6f4 failed: "
                    f"min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            "mfma-f8f6f4-e2e.mlir",
            KERNEL_NAME,
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    TestMfmaF8f6f4().test_mfma_f8f6f4_all_256()
