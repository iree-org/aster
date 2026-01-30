"""Unit tests for maybe_store_c_fragment from conditional-copies.mlir."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestMaybeStoreCFragment:
    """Test the maybe_store_c_fragment library function.

    Tests that C fragments are stored only at the last K iteration (k == K-1 AND kk ==
    KK-1).
    """

    def test_store_at_last_k_iteration(self):
        """Test that C fragments are stored only at k=K-1, kk=KK-1."""
        self._run_test(mm=2, nn=2, k=2, kk=2)

    def _run_test(self, mm: int, nn: int, k: int, kk: int):
        """Run test with configurable tile dimensions."""
        rows = mm * 16
        cols = nn * 16
        global_stride_bytes = cols * 4  # f32 = 4 bytes

        def preprocess(mlir: str) -> str:
            return (
                mlir.replace("{{MM}}", str(mm))
                .replace("{{NN}}", str(nn))
                .replace("{{K}}", str(k))
                .replace("{{KK}}", str(kk))
                .replace("{{GLOBAL_STRIDE_BYTES}}", str(global_stride_bytes))
            )

        # Output: rows x cols matrix of int32
        output = np.zeros(rows * cols, dtype=np.int32)

        compile_and_run(
            "test_maybe_store_c_fragment.mlir",
            "test_maybe_store_c_fragment",
            [],
            output,
            preprocess=preprocess,
        )

        output_2d = output.reshape(rows, cols)

        # Build expected output: mm x nn tiles of 16x16, each with MFMA C access pattern
        # MFMA 16x16x16 C fragment: lane i owns rows [i%16], cols [4*(i//16) : 4*(i//16)+4]
        expected = self._build_expected_output(mm, nn)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print("Output:")
            print(output_2d)
            print("\nExpected:")
            print(expected)

        np.testing.assert_array_equal(output_2d, expected)

    def _build_expected_output(self, mm: int, nn: int) -> np.ndarray:
        """Build expected output for mm x nn tiles with MFMA C access pattern.

        MFMA 16x16x16 C fragment layout:
        - Rows are grouped into 4 groups of 4 rows each
        - Within each group of 4 rows, all rows have identical values
        - Row group i (rows 4*i to 4*i+3) contains lanes 16*i to 16*i+15
        - Each column contains the lane index (0-15 for cols 0-15 within a tile)
        """
        rows = mm * 16
        cols = nn * 16
        expected = np.zeros((rows, cols), dtype=np.int32)

        for tile_m in range(mm):
            for tile_n in range(nn):
                for row_group in range(4):  # 4 groups of 4 rows
                    lane_base = row_group * 16
                    for row_in_group in range(4):
                        row = tile_m * 16 + row_group * 4 + row_in_group
                        for col in range(16):
                            expected[row, tile_n * 16 + col] = lane_base + col

        return expected


if __name__ == "__main__":
    TestMaybeStoreCFragment().test_store_at_last_k_iteration()
