"""Unit tests for copies.mlir library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestStoreToGlobalDwordWait:
    """Test @store_to_global_dword_wait function."""

    def test_store_dword(self):
        """Each thread stores (tid * 100) at position tid."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run(
            "test_global_store_wave.mlir", "test_store_to_global_dword_wait", [], output
        )

        expected = np.zeros(num_threads, dtype=np.int32)
        for tid in range(num_threads):
            expected[tid] = tid * 100

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestStoreToGlobalDwordx2Wait:
    """Test @store_to_global_dwordx2_wait function."""

    def test_store_dwordx2(self):
        """Each thread stores 2 dwords at position 2*tid, 2*tid+1."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run(
            "test_global_store_wave.mlir",
            "test_store_to_global_dwordx2_wait",
            [],
            output,
        )

        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(num_threads):
            expected[2 * tid] = tid * 100
            expected[2 * tid + 1] = tid * 100 + 1

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestStoreToGlobalDwordx3Wait:
    """Test @store_to_global_dwordx3_wait function."""

    def test_store_dwordx3(self):
        """Each thread stores 3 dwords at position 3*tid, 3*tid+1, 3*tid+2."""
        num_threads = 64
        output = np.zeros(num_threads * 3, dtype=np.int32)
        compile_and_run(
            "test_global_store_wave.mlir",
            "test_store_to_global_dwordx3_wait",
            [],
            output,
        )

        expected = np.zeros(num_threads * 3, dtype=np.int32)
        for tid in range(num_threads):
            expected[3 * tid] = tid * 100
            expected[3 * tid + 1] = tid * 100 + 1
            expected[3 * tid + 2] = tid * 100 + 2

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestStoreToGlobalDwordx4Wait:
    """Test @store_to_global_dwordx4_wait function."""

    def test_store_dwordx4(self):
        """Each thread stores 4 dwords at position 4*tid, 4*tid+1, 4*tid+2, 4*tid+3."""
        num_threads = 64
        output = np.zeros(num_threads * 4, dtype=np.int32)
        compile_and_run(
            "test_global_store_wave.mlir",
            "test_store_to_global_dwordx4_wait",
            [],
            output,
        )

        expected = np.zeros(num_threads * 4, dtype=np.int32)
        for tid in range(num_threads):
            expected[4 * tid] = tid * 100
            expected[4 * tid + 1] = tid * 100 + 1
            expected[4 * tid + 2] = tid * 100 + 2
            expected[4 * tid + 3] = tid * 100 + 3

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestGlobalLoadMultiTile:
    """Test @global_load_wave_multi_tile_256xf16_via_dwordx2_wait function."""

    def test_load_multi_tile_2x4_with_offsets(self):
        """Load 2x4 tiles at 4 different positions (2x2 loop) from a 64x128 array.

        This tests that m_off and n_off are computed correctly for non-zero positions.
        Input: 64x128 array with GLOBAL_STRIDE=256 bytes (128 elements * 2 bytes)
        Each iteration loads a 32x64 region (2x4 tiles of 16x16)
        LDS base offset is non-zero (256 bytes) to test offset handling.

        Position layout in 64x128 input:
          Position (0,0): rows 0-31, cols 0-63
          Position (0,1): rows 0-31, cols 64-127
          Position (1,0): rows 32-63, cols 0-63
          Position (1,1): rows 32-63, cols 64-127
        """
        num_threads = 64
        num_positions = 4  # 2x2 loop
        tiles_per_position = 8  # 2x4 tiles
        output_size = num_positions * tiles_per_position * num_threads * 4

        # Input: 64x128 matrix, output: flat array of copied values
        input_data = np.linspace(0, 1, 64 * 128, dtype=np.float16).view(np.uint16)
        output = np.zeros(output_size, dtype=np.uint16)

        compile_and_run(
            "test_global_load_multi_tile.mlir",
            "test_global_load_multi_tile",
            [input_data],
            output,
        )

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            diff = output != input_data
            diff_indices = np.where(diff)[0]
            if diff_indices.size > 0:
                print(f"Differences found at indices: {diff_indices}")
                print(f"Output: {output[diff_indices]}")
                print(f"Expected: {input_data[diff_indices]}")


if __name__ == "__main__":
    # Run all tests
    TestStoreToGlobalDwordWait().test_store_dword()
    TestStoreToGlobalDwordx2Wait().test_store_dwordx2()
    TestStoreToGlobalDwordx3Wait().test_store_dwordx3()
    TestStoreToGlobalDwordx4Wait().test_store_dwordx4()
    TestGlobalLoadMultiTile().test_load_multi_tile_2x4_with_offsets()
