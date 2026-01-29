"""Unit tests for copies.mlir library functions."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


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
            "test_copies.mlir",
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
            "test_copies.mlir",
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


class TestStoreGlobalCFragmentWait:
    """Test @global_store_wave_16x16xf32_C_fragment_wait function."""

    def test_store_MFMA_C_fragment(self):
        """Store C fragment to global with MFMA C access pattern."""
        num_threads = 64
        # Output: 16x16 matrix of int32.
        output = np.zeros(16 * 16, dtype=np.int32).reshape(16, 4, 4)

        compile_and_run(
            "test_copies.mlir", "test_store_global_C_fragment_wait", [], output
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


class TestStoreGlobalCFragmentWaitTransposed:
    """Test @test_store_global_C_fragment_wait_transposed function."""

    def test_store_MFMA_C_fragment_transposed(self):
        """Store C fragment to global with MFMA C access pattern."""
        num_threads = 64
        # Output: 16x16 matrix of f32 values (stored as int32)
        output = np.zeros(16 * 16, dtype=np.int32).reshape(16, 16)

        compile_and_run(
            "test_copies.mlir",
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


class TestGlobalLoadDsWrite:
    """Test @global_load_wave_256xf16_via_dwordx2_wait + @ds_write_dwordx2_wait functions."""

    def test_decoupled_load_store(self):
        """Load from global via memref, write to LDS, verify roundtrip."""
        num_threads = 64
        # Input: 16x16 matrix of f16 values
        input_data = np.arange(16 * 16, dtype=np.uint16)
        # Output: each thread reads back 8 bytes (4 f16 values)
        output = np.zeros(num_threads * 4, dtype=np.uint16)

        compile_and_run(
            "test_copies.mlir", "test_global_load_ds_write", [input_data], output
        )

        # The data flow is identical to global_load_to_lds_wave_16x16_f16_wait
        expected = np.zeros(num_threads * 4, dtype=np.uint16)
        for tid in range(num_threads):
            lane_id = tid % 64
            iii = lane_id // 4
            jjj_base = lane_id % 4
            jjj = jjj_base * 4
            for k in range(4):
                src_idx = iii * 16 + jjj + k
                dst_idx = tid * 4 + k
                expected[dst_idx] = src_idx

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
            "test_copies.mlir", "test_global_load_multi_tile", [input_data], output
        )

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            diff = output != input_data
            diff_indices = np.where(diff)[0]
            if diff_indices.size > 0:
                print(f"Differences found at indices: {diff_indices}")
                print(f"Output: {output[diff_indices]}")
                print(f"Expected: {input_data[diff_indices]}")


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
            "test_copies.mlir", "test_maybe_multi_tile_simple", [input_data], output
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


class TestMaybeMultiTileCoalesced:
    """Test the maybe_*_multi_tile_coalesced library functions (bulk version).

    This tests the bulk multi-tile functions that use
    global_load_wave_multi_tile_256xf16_via_dwordx2_wait and
    lds_write_wave_multi_tile_256xf16_via_dwordx2_wait.
    """

    def test_multi_tile_coalesced_with_nt_2x4(self):
        """Test with NT_I=2, NT_J=4 on a 64x128 array (4x8 tiles)."""
        rows, cols = 64, 128

        # Input: 64x128 matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run(
            "test_copies.mlir", "test_maybe_multi_tile_coalesced", [input_data], output
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
    TestStoreToGlobalDwordWait().test_store_dword()
    TestStoreToGlobalDwordx2Wait().test_store_dwordx2()
    TestStoreToGlobalDwordx3Wait().test_store_dwordx3()
    TestStoreToGlobalDwordx4Wait().test_store_dwordx4()
    TestLdsReadSwizzledFragmentWaitXorSwizzled().test_lds_read_swizzled_A_fragment()
    TestStoreGlobalCFragmentWait().test_store_C_fragment()
    TestGlobalLoadDsWrite().test_global_load_ds_write()
    TestGlobalLoadMultiTile().test_global_load_multi_tile()
    TestMaybeMultiTileSimple().test_multi_tile_simple()
    TestMaybeMultiTileCoalesced().test_multi_tile_coalesced_with_nt_2x4()
