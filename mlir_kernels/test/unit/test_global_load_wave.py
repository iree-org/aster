"""Unit tests for global_load_wave library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestGlobalLoadWave:
    """Test @global_load_wave_256xf16_via_dwordx2_wait function."""

    def test_global_load_ds_write(self):
        input = np.arange(1024, dtype=np.uint8)

        output_vx1 = np.zeros(256, dtype=np.uint8)
        output_vx2 = np.zeros(512, dtype=np.uint8)
        output_vx3 = np.zeros(768, dtype=np.uint8)
        output_vx4 = np.zeros(1024, dtype=np.uint8)

        compile_and_run(
            "test_global_load_wave.mlir",
            "test_global_load_wave",
            [input],
            [output_vx1, output_vx2, output_vx3, output_vx4],
        )

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output_vx1, input[:256])
            np.testing.assert_array_equal(output_vx2, input[:512])
            np.testing.assert_array_equal(output_vx3, input[:768])
            np.testing.assert_array_equal(output_vx4, input[:1024])


if __name__ == "__main__":
    # Run a specific test for debugging
    TestGlobalLoadWave().test_global_load_ds_write()
