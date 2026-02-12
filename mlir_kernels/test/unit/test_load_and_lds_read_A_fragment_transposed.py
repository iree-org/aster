"""Unit tests for copies.mlir library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestLoadAndReadLdsAFragmentWaitTransposed:
    """Test @test_load_and_lds_read_A_wave_16x16xf16_fragment_transposed_wait function."""

    def test_read_mfma_A_fragment(self):
        """Read A fragment from LDS with MFMA A access pattern."""
        num_threads = 64
        # Input: 16x16 matrix of f16 values (stored as uint16)
        # Each element is its linear index
        input_data = np.arange(16 * 16, dtype=np.uint16)
        # Output: each thread reads 8 bytes (4 f16 values = dwordx2)
        output = np.zeros(num_threads * 4, dtype=np.uint16)

        compile_and_run(
            "test_load_and_lds_read_A_fragment_transposed.mlir",
            "test_load_and_lds_read_A_wave_16x16xf16_fragment_transposed_wait",
            [input_data],
            output,
        )

        # Verify: mfma_index_A returns (lane_id % 16, 4 * (lane_id // 16))
        # So row = lane_id % 16, col_base = 4 * (lane_id // 16)
        # Each thread reads 4 consecutive f16 values
        expected = np.zeros(num_threads * 4, dtype=np.uint16)
        for tid in range(num_threads):
            lane_id = tid % 64
            # mfma_index_A transposed: ii = 4 * (lane_id // 16), jj = lane_id % 16
            ii = 4 * (lane_id // 16)
            jj = lane_id % 16
            # Each thread reads 4 consecutive values starting at (ii, jj)
            for k in range(4):
                src_idx = ii * 16 + jj + k
                dst_idx = tid * 4 + k
                expected[dst_idx] = src_idx

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(
                output.reshape(16, 16), expected.reshape(16, 16)
            )


if __name__ == "__main__":
    # Run all tests
    TestLoadAndReadLdsAFragmentWaitTransposed().test_read_mfma_A_fragment()
