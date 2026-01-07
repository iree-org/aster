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


class TestGlobalLoadMultiTile:
    """Test @global_load_wave_multi_tile_256xf16_via_dwordx2_wait function."""

    def test_load_multi_tile_2x3(self):
        """Load 2x3 tiles (32x48 region = six 16x16 tiles) from global memory."""
        num_threads = 64
        # Input: 32x48 matrix of f16 values (stored as uint16)
        # Layout: row-major with 48 elements per row
        input_data = np.arange(32 * 48, dtype=np.uint16)
        # Output: each thread reads 4 f16 values from each tile
        # 6 tiles * 64 threads * 4 values = 1536 values total
        output = np.zeros(num_threads * 4 * 6, dtype=np.uint16)

        compile_and_run(
            "test_global_load_multi_tile", [input_data], output
        )

        # Expected output:
        # Tile (0,0): columns 0-15, rows 0-15 -> output[0:256]
        # Tile (0,1): columns 16-31, rows 0-15 -> output[256:512]
        # Tile (0,2): columns 32-47, rows 0-15 -> output[512:768]
        # Tile (1,0): columns 0-15, rows 16-31 -> output[768:1024]
        # Tile (1,1): columns 16-31, rows 16-31 -> output[1024:1280]
        # Tile (1,2): columns 32-47, rows 16-31 -> output[1280:1536]
        expected = np.zeros(num_threads * 4 * 6, dtype=np.uint16)

        for tid in range(num_threads):
            lane_id = tid % 64
            # lane_delinearize_2d(16, 4): iii = lane_id // 4, jjj = lane_id % 4
            iii = lane_id // 4
            jjj = lane_id % 4
            jjj_pos = jjj * 4

            # Tile (0,0): columns 0-15, rows 0-15
            for k in range(4):
                src_idx = iii * 48 + jjj_pos + k  # row * stride + col
                dst_idx = tid * 4 + k
                expected[dst_idx] = src_idx

            # Tile (0,1): columns 16-31, rows 0-15
            for k in range(4):
                src_idx = iii * 48 + 16 + jjj_pos + k  # row * stride + 16 + col
                dst_idx = num_threads * 4 + tid * 4 + k
                expected[dst_idx] = src_idx

            # Tile (0,2): columns 32-47, rows 0-15
            for k in range(4):
                src_idx = iii * 48 + 32 + jjj_pos + k  # row * stride + 32 + col
                dst_idx = num_threads * 4 * 2 + tid * 4 + k
                expected[dst_idx] = src_idx

            # Tile (1,0): columns 0-15, rows 16-31
            for k in range(4):
                src_idx = (iii + 16) * 48 + jjj_pos + k  # (row + 16) * stride + col
                dst_idx = num_threads * 4 * 3 + tid * 4 + k
                expected[dst_idx] = src_idx

            # Tile (1,1): columns 16-31, rows 16-31
            for k in range(4):
                src_idx = (iii + 16) * 48 + 16 + jjj_pos + k  # (row + 16) * stride + 16 + col
                dst_idx = num_threads * 4 * 4 + tid * 4 + k
                expected[dst_idx] = src_idx

            # Tile (1,2): columns 32-47, rows 16-31
            for k in range(4):
                src_idx = (iii + 16) * 48 + 32 + jjj_pos + k  # (row + 16) * stride + 32 + col
                dst_idx = num_threads * 4 * 5 + tid * 4 + k
                expected[dst_idx] = src_idx

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    # Run a specific test for debugging
    TestLoadAndReadLdsAFragmentWait().test_read_A_fragment_swizzled()
