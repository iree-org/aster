"""Unit tests for loop execution."""

import numpy as np
from aster.pass_pipelines import TEST_LOOP_PASS_PIPELINE

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestUniformLoopLowering:
    """Test uniform loop lowering."""

    def test_uniform_loop(self):
        """The output buffer should contain [0, ..., n - 1] * 4 after execution."""
        num_threads = 64
        output = np.zeros(64, dtype=np.int32)
        compile_and_run(
            "test_loops.mlir",
            "test_uniform_loop",
            # TODO: properly handle int arguments
            input_data=[np.array([output.size], dtype=np.int32)],
            output_data=output,
            block_dim=(num_threads, 1, 1),
            pass_pipeline=TEST_LOOP_PASS_PIPELINE,
            library_paths=[],  # No library needed for this test
        )
        expected = np.arange(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected * 4)


if __name__ == "__main__":
    TestUniformLoopLowering().test_uniform_loop()
