import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def test_readfirstlane_const():
    """v_readfirstlane_b32 on a constant: every lane should get 42."""
    block = (64, 1, 1)
    output = np.zeros(64, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.full(64, 42, dtype=np.int32)
        np.testing.assert_array_equal(
            outputs[0],
            expected,
            err_msg="all lanes should read 42 from readfirstlane",
        )

    compile_and_run(
        "readfirstlane-e2e.mlir",
        "readfirstlane_const",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


def test_readfirstlane_tid():
    """v_readfirstlane_b32 on tid_x: lane 0's tid_x is 0, broadcast to all."""
    block = (64, 1, 1)
    output = np.zeros(64, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.zeros(64, dtype=np.int32)
        np.testing.assert_array_equal(
            outputs[0],
            expected,
            err_msg="all lanes should read 0 (lane 0's tid_x) from readfirstlane",
        )

    compile_and_run(
        "readfirstlane-e2e.mlir",
        "readfirstlane_tid",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
