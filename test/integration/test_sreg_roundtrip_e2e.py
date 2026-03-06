"""E2E test: write 42 to M0, read back to SGPR, broadcast to VGPR, store to global buffer."""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

MCPU = "gfx942"
TOTAL_LANES = 64


def test_sreg_roundtrip():
    """All 64 lanes should store 42 to their output slot."""
    output = np.zeros(TOTAL_LANES, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.full(TOTAL_LANES, 42, dtype=np.int32)
        np.testing.assert_array_equal(
            outputs[0],
            expected,
            err_msg="All lanes should contain 42",
        )

    compile_and_run(
        "sreg-roundtrip-e2e.mlir",
        "m0_roundtrip_kernel",
        input_data=[],
        output_data=[output],
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=TOTAL_LANES,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
