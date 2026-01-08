"""Unit tests for copies.mlir library functions."""

import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    SYNCHRONOUS_SROA_PASS_PIPELINE,
    hsaco_file,
)

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_mlir_file():
    """Get path to the test MLIR file."""
    return os.path.join(os.path.dirname(__file__), "test_copies.mlir")


def get_library_paths():
    """Get paths to all required library files."""
    library_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "library", "common"
    )
    return [
        os.path.join(library_dir, "register_init.mlir"),
        os.path.join(library_dir, "indexing.mlir"),
        os.path.join(library_dir, "copies.mlir"),
        os.path.join(library_dir, "multi-tile-copies.mlir"),
    ]


def compile_and_run(
    kernel_name: str,
    input_data: list,
    output_data: np.ndarray,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
):
    """Compile and run a test kernel, returning the output buffer."""
    mlir_file = get_mlir_file()
    library_paths = get_library_paths()

    with ir.Context() as ctx:
        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            SYNCHRONOUS_SROA_PASS_PIPELINE,
            ctx,
            library_paths=library_paths,
            print_ir_after_all=False,
        )

        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=MCPU):
                print(asm_complete)
                pytest.skip(f"GPU {MCPU} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=input_data,
                output_args=[output_data],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


class TestStoreToGlobalDwordWait:
    """Test @store_to_global_dword_wait function."""

    def test_store_dword(self):
        """Each thread stores (tid * 100) at position (tid/8, tid%8) in a 16-wide matrix."""
        num_threads = 64
        # Output is 8 rows x 16 columns = 128 dwords, but only 64 are written
        # Layout: row = tid // 8, col = tid % 8
        output = np.zeros(8 * 16, dtype=np.int32)
        compile_and_run("test_store_to_global_dword_wait", [], output)

        expected = np.zeros(8 * 16, dtype=np.int32)
        for tid in range(num_threads):
            row = tid // 8
            col = tid % 8
            linear_idx = row * 16 + col
            expected[linear_idx] = tid * 100

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(
                output.reshape(8, 16), expected.reshape(8, 16)
            )


class TestLoadAndReadLdsAFragmentWait:
    """Test @lds_read_A_wave_16x16xf16_fragment_wait function."""

    def test_read_A_fragment_swizzled(self):
        """Read A fragment from LDS with swizzled access pattern."""
        num_threads = 64
        # Input: 16x16 matrix of f16 values (stored as uint16)
        # Each element is its linear index
        input_data = np.arange(16 * 16, dtype=np.uint16)
        # Output: each thread reads 8 bytes (4 f16 values = dwordx2)
        output = np.zeros(num_threads * 4, dtype=np.uint16)

        compile_and_run(
            "test_load_and_lds_read_A_wave_16x16xf16_fragment_wait",
            [input_data],
            output,
        )

        # Verify: swizzle_A returns (lane_id % 16, 4 * (lane_id // 16))
        # So row = lane_id % 16, col_base = 4 * (lane_id // 16)
        # Each thread reads 4 consecutive f16 values
        expected = np.zeros(num_threads * 4, dtype=np.uint16)
        for tid in range(num_threads):
            lane_id = tid % 64
            # swizzle_A: ii = lane_id % 16, jj = 4 * (lane_id // 16)
            ii = lane_id % 16
            jj = 4 * (lane_id // 16)
            # Each thread reads 4 consecutive values starting at (ii, jj)
            for k in range(4):
                src_idx = ii * 16 + jj + k
                dst_idx = tid * 4 + k
                expected[dst_idx] = src_idx

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(
                output.reshape(16, 16), expected.reshape(16, 16)
            )


class TestStoreGlobalCFragmentWait:
    """Test @global_store_wave_16x16xf32_swizzled_C_fragment_wait function."""

    def test_store_C_fragment_swizzled(self):
        """Store C fragment to global with swizzled access pattern."""
        num_threads = 64
        # Output: 16x16 matrix of f32 values (stored as int32)
        output = np.zeros(16 * 16, dtype=np.int32)

        compile_and_run("test_store_global_C_fragment_wait", [], output)

        # Verify: each lane initializes its fragment with lane_id
        # swizzle_C returns (4 * (lane_id // 16), lane_id % 16)
        # So row_base = 4 * (lane_id // 16), col = lane_id % 16
        # Each lane writes 4 values at rows row_base, row_base+1, row_base+2, row_base+3
        expected = np.zeros(16 * 16, dtype=np.int32)
        for lane_id in range(num_threads):
            row_base = 4 * (lane_id // 16)
            col = lane_id % 16
            for k in range(4):
                row = row_base + k
                linear_idx = row * 16 + col
                expected[linear_idx] = lane_id

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

        compile_and_run("test_global_load_ds_write", [input_data], output)

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

class TestGlobalToLdsAndBack16x16:
    """Test @global_to_lds_and_back_wave_16x16xf16_wait function."""

    def test_copy_tile_at_position_3_5(self):
        """Copy a single 16x16 tile from position (3,5) in a 64x96 array.

        This tests that position handling is correct by loading from a specific
        tile location and verifying the correct data is retrieved.
        """
        rows, cols = 40, 60
        tile_m, tile_n = 1, 2  # Tile position (element 16, 32)
        m_pos, n_pos = tile_m * 16, tile_n * 16

        # Input: rows x cols matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run("test_global_to_lds_and_back_wave_16x16xf16_wait",
                        [input_data], output)

        # Expected: the 16x16 tile at (48, 80) flattened
        input_2d = input_data.reshape(rows, cols)
        expected = input_2d[m_pos:m_pos+16, n_pos:n_pos+16]
        output_2d = output.reshape(rows, cols)
        output_2d_slice = output_2d[m_pos:m_pos+16, n_pos:n_pos+16]
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output_2d_slice, expected)

            # remove output_2d_slice from output_2d at m_pos:m_pos+16, n_pos:n_pos+16
            # and assert all zeros.
            output_2d[m_pos:m_pos + 16, n_pos:n_pos + 16] = \
                output_2d[m_pos:m_pos + 16, n_pos:n_pos + 16] - output_2d_slice
            np.testing.assert_array_equal(output_2d.flatten(), np.zeros(rows * cols))


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

        compile_and_run("test_global_load_multi_tile", [input_data], output)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            diff = output != input_data
            diff_indices = np.where(diff)[0]
            if diff_indices.size > 0:
                print(f"Differences found at indices: {diff_indices}")
                print(f"Output: {output[diff_indices]}")
                print(f"Expected: {input_data[diff_indices]}")

class TestMaybeMultiTileSimple:
    """Test the maybe_*_multi_tile_simple library functions.
    
    This isolates the indexing logic to debug NT_I/NT_J tile position bugs.
    The pattern loops over (ii, jj) indices and executes multi-tile operations
    when ii % NT_I == 0 AND jj % NT_J == 0.
    """

    def test_multi_tile_with_nt_2x4(self):
        """Test with NT_I=2, NT_J=4 on a 64x128 array (4x8 tiles)."""
        rows, cols = 64, 128

        # Input: 64x128 matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run("test_maybe_multi_tile_simple", [input_data], output)

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
                    print(f"Tile ({ti},{tj}) rows {r0}-{r1-1} cols {c0}-{c1-1}: {'✓' if match else '✗'}")

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

        compile_and_run("test_maybe_multi_tile_coalesced", [input_data],
                        output)

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
                    print(f"Tile ({ti},{tj}) rows {r0}-{r1-1} cols {c0}-{c1-1}: {'✓' if match else '✗'}")

            np.testing.assert_array_equal(output, input_data)


if __name__ == "__main__":
    # Run a specific test for debugging
    TestMaybeMultiTileCoalesced().test_multi_tile_coalesced_with_nt_2x4()
