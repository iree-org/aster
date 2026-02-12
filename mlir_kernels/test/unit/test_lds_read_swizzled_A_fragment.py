"""Unit tests for copies.mlir library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestLdsReadSwizzledFragmentWaitXorSwizzled:
    """Test @lds_read_A_wave_16x16xf16_fragment_wait function with XOR swizzling."""

    def test_read_mfma_A_fragment_xor_swizzled(self):
        """Read A fragment from LDS with XOR-swizzled MFMA A access pattern.

        Tests 2x3 tiles of 16x16, each tile contains iota 0-255.
        """
        num_threads = 64
        II, JJ = 2, 3  # 2x3 tiles

        # Input: 32x48 matrix (2x3 tiles of 16x16) in 2D row-major layout.
        # Each 16x16 tile contains iota 0-255.
        tile_iota = np.arange(256, dtype=np.uint16).reshape(16, 16)
        input_2d = np.zeros((II * 16, JJ * 16), dtype=np.uint16)
        for ii in range(II):
            for jj in range(JJ):
                input_2d[ii * 16 : (ii + 1) * 16, jj * 16 : (jj + 1) * 16] = tile_iota

        # Output: 32x48 matrix
        output = np.zeros(II * JJ * 16 * 16, dtype=np.uint16)

        compile_and_run(
            "test_lds_read_swizzled_A_fragment.mlir",
            "test_lds_read_swizzled_A_wave_16x16xf16_fragment_wait",
            [input_2d.flatten()],
            output,
        )

        # Compute expected values for each tile.
        # Each tile should produce the same swizzled pattern since each tile has
        # the same iota values.
        # fmt: off
        expected = np.array([
            [  0,   1,   2,   3,  16,  17,  18,  19,  32,  33,  34,  35,  48,  49,  50,  51],
            [ 68,  69,  70,  71,  84,  85,  86,  87, 100, 101, 102, 103, 116, 117, 118, 119],
            [136, 137, 138, 139, 152, 153, 154, 155, 168, 169, 170, 171, 184, 185, 186, 187],
            [204, 205, 206, 207, 220, 221, 222, 223, 236, 237, 238, 239, 252, 253, 254, 255],
            [  4,   5,   6,   7,  20,  21,  22,  23,  36,  37,  38,  39,  52,  53,  54,  55],
            [ 64,  65,  66,  67,  80,  81,  82,  83,  96,  97,  98,  99, 112, 113, 114, 115],
            [140, 141, 142, 143, 156, 157, 158, 159, 172, 173, 174, 175, 188, 189, 190, 191],
            [200, 201, 202, 203, 216, 217, 218, 219, 232, 233, 234, 235, 248, 249, 250, 251],
            [  8,   9,  10,  11,  24,  25,  26,  27,  40,  41,  42,  43,  56,  57,  58,  59],
            [ 76,  77,  78,  79,  92,  93,  94,  95, 108, 109, 110, 111, 124, 125, 126, 127],
            [128, 129, 130, 131, 144, 145, 146, 147, 160, 161, 162, 163, 176, 177, 178, 179],
            [196, 197, 198, 199, 212, 213, 214, 215, 228, 229, 230, 231, 244, 245, 246, 247],
            [ 12,  13,  14,  15,  28,  29,  30,  31,  44,  45,  46,  47,  60,  61,  62,  63],
            [ 72,  73,  74,  75,  88,  89,  90,  91, 104, 105, 106, 107, 120, 121, 122, 123],
            [132, 133, 134, 135, 148, 149, 150, 151, 164, 165, 166, 167, 180, 181, 182, 183],
            [192, 193, 194, 195, 208, 209, 210, 211, 224, 225, 226, 227, 240, 241, 242, 243],
        ], dtype=np.uint16)
        # fmt: on

        # Reshape output to 2D for tile extraction
        output_2d = output.reshape(II * 16, JJ * 16)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            # Check each tile separately for better error messages
            for ii in range(II):
                for jj in range(JJ):
                    input_tile = input_2d[
                        ii * 16 : (ii + 1) * 16, jj * 16 : (jj + 1) * 16
                    ]
                    output_tile = output_2d[
                        ii * 16 : (ii + 1) * 16, jj * 16 : (jj + 1) * 16
                    ]
                    np.testing.assert_array_equal(
                        output_tile, expected, err_msg=f"Mismatch at tile ({ii}, {jj})"
                    )


if __name__ == "__main__":
    # Run all tests
    TestLdsReadSwizzledFragmentWaitXorSwizzled().test_read_mfma_A_fragment_xor_swizzled()
