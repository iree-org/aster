"""E2E test: pass a scalar i32 via by_val_arg, store to output[tid_x]."""

import numpy as np

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

MCPU = "gfx942"
TOTAL_LANES = 64


def test_by_val_store():
    """All 64 lanes should store the by_val scalar to their output slot."""
    scalar = 0xDEAD
    output = np.zeros(TOTAL_LANES, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.full(TOTAL_LANES, scalar, dtype=np.int32)
        np.testing.assert_array_equal(outputs[0], expected)

    compile_and_run(
        "by-val-store-e2e.mlir",
        "by_val_store",
        input_data=[scalar],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )
