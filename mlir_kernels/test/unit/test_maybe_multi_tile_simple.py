"""Unit tests for maybe_simple_global_load_wave_multi_tile_16x16xf16 and maybe_simple_lds_write_wave_multi_tile_16x16xf16."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestMaybeMultiTileSimple:
    """Test the maybe_*_multi_tile_simple library functions.

    This isolates the indexing logic to debug NT_I/NT_J tile position bugs. The pattern
    loops over (ii, jj) indices and executes multi-tile operations when ii % NT_I == 0
    AND jj % NT_J == 0.
    """

    def test_multi_tile_with_nt_2x4(self):
        """Test with NT_I=2, NT_J=4 on a 64x128 array (4x8 tiles)."""
        rows, cols = 64, 128

        # Input: 64x128 matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run(
            "test_maybe_multi_tile_simple.mlir",
            "test_maybe_multi_tile_simple",
            [input_data],
            output,
        )

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            input_2d = input_data.reshape(rows, cols)
            output_2d = output.reshape(rows, cols)

            # Check which 16x16 tiles have correct data
            for ti in range(4):
                for tj in range(8):
                    r0, r1 = ti * 16, (ti + 1) * 16
                    c0, c1 = tj * 16, (tj + 1) * 16
                    tile_in = input_2d[r0:r1, c0:c1]
                    tile_out = output_2d[r0:r1, c0:c1]
                    match = np.array_equal(tile_in, tile_out)
                    print(
                        f"Tile ({ti},{tj}) rows {r0}-{r1-1} cols {c0}-{c1-1}: {'✓' if match else '✗'}"
                    )

            np.testing.assert_array_equal(output, input_data)


if __name__ == "__main__":
    # Run all tests
    TestMaybeMultiTileSimple().test_multi_tile_with_nt_2x4()
