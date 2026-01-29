"""Unit tests for copies.mlir library functions."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestStoreGlobalCFragmentWaitTransposed:
    """Test @test_store_global_C_fragment_wait_transposed function."""

    def test_store_MFMA_C_fragment_transposed(self):
        """Store C fragment to global with MFMA C access pattern."""
        num_threads = 64
        # Output: 16x16 matrix of f32 values (stored as int32)
        output = np.zeros(16 * 16, dtype=np.int32).reshape(16, 16)

        compile_and_run(
            "test_store_global_C_fragment_transposed.mlir",
            "test_store_global_C_fragment_wait_transposed",
            [],
            output,
        )

        # fmt: off
        expected = np.array([
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
        ], dtype=np.int32)
        # fmt: on

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    # Run all tests
    TestStoreGlobalCFragmentWaitTransposed().test_store_MFMA_C_fragment_transposed()
