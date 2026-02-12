"""Unit tests for copies.mlir library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestStoreGlobalCFragmentWait:
    """Test @global_store_wave_16x16xf32_C_fragment_wait function."""

    def test_store_MFMA_C_fragment(self):
        """Store C fragment to global with MFMA C access pattern."""
        num_threads = 64
        # Output: 16x16 matrix of int32.
        output = np.zeros(16 * 16, dtype=np.int32).reshape(16, 4, 4)

        compile_and_run(
            "test_store_global_C_fragment.mlir",
            "test_store_global_C_fragment_wait",
            [],
            output,
        )

        # fmt: off
        expected = np.array([
            [[0] * 4, [16] * 4, [32] * 4, [48] * 4],
            [[1] * 4, [17] * 4, [33] * 4, [49] * 4],
            [[2] * 4, [18] * 4, [34] * 4, [50] * 4],
            [[3] * 4, [19] * 4, [35] * 4, [51] * 4],
            [[4] * 4, [20] * 4, [36] * 4, [52] * 4],
            [[5] * 4, [21] * 4, [37] * 4, [53] * 4],
            [[6] * 4, [22] * 4, [38] * 4, [54] * 4],
            [[7] * 4, [23] * 4, [39] * 4, [55] * 4],
            [[8] * 4, [24] * 4, [40] * 4, [56] * 4],
            [[9] * 4, [25] * 4, [41] * 4, [57] * 4],
            [[10] * 4, [26] * 4, [42] * 4, [58] * 4],
            [[11] * 4, [27] * 4, [43] * 4, [59] * 4],
            [[12] * 4, [28] * 4, [44] * 4, [60] * 4],
            [[13] * 4, [29] * 4, [45] * 4, [61] * 4],
            [[14] * 4, [30] * 4, [46] * 4, [62] * 4],
            [[15] * 4, [31] * 4, [47] * 4, [63] * 4],
        ], dtype=np.int32)
        # fmt: on

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print(output)
            print(expected)
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    # Run all tests
    TestStoreGlobalCFragmentWait().test_store_MFMA_C_fragment()
