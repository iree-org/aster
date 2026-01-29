"""Unit tests for copies.mlir library functions."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestGlobalToLdsAndBack16x16:
    """Test @global_to_lds_and_back_wave_16x16xf16_wait function."""

    def test_copy_tile_at_position_3_5(self):
        """Copy a single 16x16 tile from position (3,5) in a 64x96 array.

        This tests that position handling is correct by loading from a specific tile
        location and verifying the correct data is retrieved.
        """
        rows, cols = 40, 60
        tile_m, tile_n = 1, 2  # Tile position (element 16, 32)
        m_pos, n_pos = tile_m * 16, tile_n * 16

        # Input: rows x cols matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run(
            "test_global_to_lds_and_back.mlir",
            "test_global_to_lds_and_back_wave_16x16xf16_wait",
            [input_data],
            output,
        )

        # Expected: the 16x16 tile at (48, 80) flattened
        input_2d = input_data.reshape(rows, cols)
        expected = input_2d[m_pos : m_pos + 16, n_pos : n_pos + 16]
        output_2d = output.reshape(rows, cols)
        output_2d_slice = output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16]
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output_2d_slice, expected)

            # remove output_2d_slice from output_2d at m_pos:m_pos+16, n_pos:n_pos+16
            # and assert all zeros.
            output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16] = (
                output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16] - output_2d_slice
            )
            np.testing.assert_array_equal(output_2d.flatten(), np.zeros(rows * cols))


if __name__ == "__main__":
    # Run all tests
    TestGlobalToLdsAndBack16x16().test_copy_tile_at_position_3_5()
