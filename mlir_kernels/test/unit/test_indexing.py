"""Unit tests for indexing.mlir library functions."""

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
    return os.path.join(os.path.dirname(__file__), "test_indexing.mlir")


def get_library_path():
    """Get path to the indexing library."""
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "library", "common", "indexing.mlir"
    )


def compile_and_run(
    kernel_name: str, output_data: np.ndarray, grid_dim=(1, 1, 1), block_dim=(64, 1, 1)
):
    """Compile and run a test kernel, returning the output buffer."""
    mlir_file = get_mlir_file()
    library_path = get_library_path()

    def preprocess(x):
        x = x.replace(
            "{{NUM_THREADS}}", str(block_dim[0] * block_dim[1] * block_dim[2])
        )
        x = x.replace("{{NUM_BLOCKS}}", str(grid_dim[0] * grid_dim[1] * grid_dim[2]))
        return x

    with ir.Context() as ctx:
        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            SYNCHRONOUS_SROA_PASS_PIPELINE,
            ctx,
            library_paths=[library_path],
            print_ir_after_all=False,
            preprocess=preprocess,
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
                input_args=[],
                output_args=[output_data],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


class TestLaneId:
    """Test @lane_id function."""

    def test_lane_id(self):
        """Each thread should output its lane_id (0..63)."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_lane_id", output)
        expected = np.arange(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestWaveId:
    """Test @wave_id function."""

    def test_wave_id_single_wave(self):
        """With one wave, all threads should output wave_id=0."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_wave_id", output)
        expected = np.zeros(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)

    def test_wave_id_two_waves(self):
        """With two waves, threads 0-63 output 0, threads 64-127 output 1."""
        num_threads = 128
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_wave_id", output, block_dim=(num_threads, 1, 1))
        expected = np.array([0] * 64 + [1] * 64, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestWaveCount:
    """Test @wave_count function."""

    def test_wave_count_single_wave(self):
        """With 64 threads, wave_count should be 1."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_wave_count", output)
        expected = np.ones(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)

    def test_wave_count_two_waves(self):
        """With 128 threads, wave_count should be 2."""
        num_threads = 128
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_wave_count", output, block_dim=(num_threads, 1, 1))
        expected = np.full(num_threads, 2, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestWavePartition2D:
    """Test @lane_delinearize_2d function."""

    def test_wave_partition_8x8(self):
        """Partition 64 lanes into 8x8 grid."""
        # Output: pairs of (i, j) at indices tid * 2, tid * 2 + 1
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_lane_delinearize_2d", output)

        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(num_threads):
            expected[tid * 2] = tid // 8
            expected[tid * 2 + 1] = tid % 8

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestGridPartition2D:
    """Test @block_id_x_delinearize_2d function."""

    def test_grid_partition_2x4(self):
        num_blocks = 8
        num_threads = 64
        output = np.zeros(num_threads * 2 * num_blocks, dtype=np.int32)
        compile_and_run(
            "test_block_id_x_delinearize_2d",
            output,
            block_dim=(num_threads, 1, 1),
            grid_dim=(num_blocks, 1, 1),
        )

        expected = np.zeros(num_threads * 2 * num_blocks, dtype=np.int32)
        for bid in range(num_blocks):
            for tid in range(num_threads):
                global_tid = bid * num_threads + tid
                expected[global_tid * 2] = bid // 4
                expected[global_tid * 2 + 1] = bid % 4

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestTiledGridPartition2D:
    """Test @tiled_grid_partition_2D function."""

    def test_tiled_grid_partition_64x64_32x32(self):
        """Partition 64x64 problem with 32x32 tiles -> 2x2 tile grid."""
        num_threads = 64
        num_blocks = 4
        output = np.zeros(num_threads * 2 * num_blocks, dtype=np.int32)
        compile_and_run(
            "test_tiled_grid_partition_2D",
            output,
            grid_dim=(num_blocks, 1, 1),
        )

        expected = np.zeros(num_threads * 2 * num_blocks, dtype=np.int32)
        for bid in range(num_blocks):
            expected_i = bid // 2
            expected_j = bid % 2
            for tid in range(64):
                global_tid = bid * 64 + tid
                expected[global_tid * 2] = expected_i
                expected[global_tid * 2 + 1] = expected_j

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMatrixOffset:
    """Test @matrix_offset function."""

    def test_matrix_offset(self):
        """Compute byte offset for 2D matrix access."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_matrix_offset", output)

        # offset = (i * N + j) * elt_size
        # i = tid / 8, j = tid % 8, N = 16, elt_size = 4
        expected = np.array([
            0, 4, 8, 12, 16, 20, 24, 28,  # tid 0-7: i=0, j=0-7
            64, 68, 72, 76, 80, 84, 88, 92,  # tid 8-15: i=1, j=0-7
            128, 132, 136, 140, 144, 148, 152, 156,  # tid 16-23: i=2, j=0-7
            192, 196, 200, 204, 208, 212, 216, 220,  # tid 24-31: i=3, j=0-7
            256, 260, 264, 268, 272, 276, 280, 284,  # tid 32-39: i=4, j=0-7
            320, 324, 328, 332, 336, 340, 344, 348,  # tid 40-47: i=5, j=0-7
            384, 388, 392, 396, 400, 404, 408, 412,  # tid 48-55: i=6, j=0-7
            448, 452, 456, 460, 464, 468, 472, 476,  # tid 56-63: i=7, j=0-7
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestTiledMatrixOffset:
    """Test @tiled_matrix_offset function."""

    def test_tiled_matrix_offset(self):
        """Compute byte offset for tiled 2D matrix access."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_tiled_matrix_offset", output)

        # offset = ((i + ii) * N + (j + jj)) * elt_size
        # i=0, j=0, ii = tid / 8, jj = tid % 8, N = 16, elt_size = 4
        expected = np.array([
            0, 4, 8, 12, 16, 20, 24, 28,  # tid 0-7: ii=0, jj=0-7
            64, 68, 72, 76, 80, 84, 88, 92,  # tid 8-15: ii=1, jj=0-7
            128, 132, 136, 140, 144, 148, 152, 156,  # tid 16-23: ii=2, jj=0-7
            192, 196, 200, 204, 208, 212, 216, 220,  # tid 24-31: ii=3, jj=0-7
            256, 260, 264, 268, 272, 276, 280, 284,  # tid 32-39: ii=4, jj=0-7
            320, 324, 328, 332, 336, 340, 344, 348,  # tid 40-47: ii=5, jj=0-7
            384, 388, 392, 396, 400, 404, 408, 412,  # tid 48-55: ii=6, jj=0-7
            448, 452, 456, 460, 464, 468, 472, 476,  # tid 56-63: ii=7, jj=0-7
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestTiledx2MatrixOffset:
    """Test @tiledx2_matrix_offset function."""

    def test_tiledx2_matrix_offset(self):
        """Compute byte offset for twice-tiled 2D matrix access."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_tiledx2_matrix_offset", output)

        # offset = ((i + ii + iii) * N + (j + jj + jjj)) * elt_size
        # i=0, j=0, ii=0, jj=0, iii = tid / 8, jjj = tid % 8, N = 16, elt_size = 4
        expected = np.array([
            0, 4, 8, 12, 16, 20, 24, 28,  # tid 0-7: iii=0, jjj=0-7
            64, 68, 72, 76, 80, 84, 88, 92,  # tid 8-15: iii=1, jjj=0-7
            128, 132, 136, 140, 144, 148, 152, 156,  # tid 16-23: iii=2, jjj=0-7
            192, 196, 200, 204, 208, 212, 216, 220,  # tid 24-31: iii=3, jjj=0-7
            256, 260, 264, 268, 272, 276, 280, 284,  # tid 32-39: iii=4, jjj=0-7
            320, 324, 328, 332, 336, 340, 344, 348,  # tid 40-47: iii=5, jjj=0-7
            384, 388, 392, 396, 400, 404, 408, 412,  # tid 48-55: iii=6, jjj=0-7
            448, 452, 456, 460, 464, 468, 472, 476,  # tid 56-63: iii=7, jjj=0-7
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndex16x16Helper:
    """Test @mfma_index_16x16_helper function."""

    def test_mfma_index_16x16_helper(self):
        """Returns (4 * (lane_id / 16), lane_id mod 16)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_16x16_helper", output)

        expected = np.array([
            0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,  # tid 0-7: lane_id 0-7
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15,  # tid 8-15: lane_id 8-15
            4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7,  # tid 16-23: lane_id 16-23
            4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14, 4, 15,  # tid 24-31: lane_id 24-31
            8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7,  # tid 32-39: lane_id 32-39
            8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15,  # tid 40-47: lane_id 40-47
            12, 0, 12, 1, 12, 2, 12, 3, 12, 4, 12, 5, 12, 6, 12, 7,  # tid 48-55: lane_id 48-55
            12, 8, 12, 9, 12, 10, 12, 11, 12, 12, 12, 13, 12, 14, 12, 15,  # tid 56-63: lane_id 56-63
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndexA16x16xf16:
    """Test @mfma_index_A_16x16xf16 function."""

    def test_mfma_index_A_16x16xf16(self):
        """MFMA indexing for A fragment (swapped from helper)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_A_16x16xf16", output)

        # A MFMA indexing returns (j, i) from helper, which is (lane_id mod 16, 4 * (lane_id / 16))
        expected = np.array([
            0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,  # tid 0-7: lane_id 0-7 -> (j, i) = (0-7, 0)
            8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,  # tid 8-15: lane_id 8-15 -> (j, i) = (8-15, 0)
            0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4,  # tid 16-23: lane_id 16-23 -> (j, i) = (0-7, 4)
            8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14, 4, 15, 4,  # tid 24-31: lane_id 24-31 -> (j, i) = (8-15, 4)
            0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7, 8,  # tid 32-39: lane_id 32-39 -> (j, i) = (0-7, 8)
            8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15, 8,  # tid 40-47: lane_id 40-47 -> (j, i) = (8-15, 8)
            0, 12, 1, 12, 2, 12, 3, 12, 4, 12, 5, 12, 6, 12, 7, 12,  # tid 48-55: lane_id 48-55 -> (j, i) = (0-7, 12)
            8, 12, 9, 12, 10, 12, 11, 12, 12, 12, 13, 12, 14, 12, 15, 12,  # tid 56-63: lane_id 56-63 -> (j, i) = (8-15, 12)
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndexB16x16xf16:
    """Test @mfma_index_B_16x16xf16 function."""

    def test_mfma_index_B_16x16xf16(self):
        """MFMA indexing for B fragment (same as helper)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_B_16x16xf16", output)

        expected = np.array([
            0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,  # tid 0-7: lane_id 0-7
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15,  # tid 8-15: lane_id 8-15
            4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7,  # tid 16-23: lane_id 16-23
            4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14, 4, 15,  # tid 24-31: lane_id 24-31
            8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7,  # tid 32-39: lane_id 32-39
            8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15,  # tid 40-47: lane_id 40-47
            12, 0, 12, 1, 12, 2, 12, 3, 12, 4, 12, 5, 12, 6, 12, 7,  # tid 48-55: lane_id 48-55
            12, 8, 12, 9, 12, 10, 12, 11, 12, 12, 12, 13, 12, 14, 12, 15,  # tid 56-63: lane_id 56-63
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndexC16x16xf32:
    """Test @mfma_index_C_16x16xf32 function."""

    def test_mfma_index_C_16x16xf32(self):
        """MFMA indexing for C fragment (same as helper)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_C_16x16xf32", output)

        expected = np.array([
            0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,  # tid 0-7: lane_id 0-7
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15,  # tid 8-15: lane_id 8-15
            4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7,  # tid 16-23: lane_id 16-23
            4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14, 4, 15,  # tid 24-31: lane_id 24-31
            8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7,  # tid 32-39: lane_id 32-39
            8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15,  # tid 40-47: lane_id 40-47
            12, 0, 12, 1, 12, 2, 12, 3, 12, 4, 12, 5, 12, 6, 12, 7,  # tid 48-55: lane_id 48-55
            12, 8, 12, 9, 12, 10, 12, 11, 12, 12, 12, 13, 12, 14, 12, 15,  # tid 56-63: lane_id 56-63
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestSwizzledMfmaIndexA16x16xf16:
    """Test @swizzled_mfma_index_A_16x16xf16 function."""

    def test_swizzled_mfma_index_A(self):
        """Swizzled MFMA indexing for A fragment (transposed + XOR swizzle)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_swizzled_mfma_index_A_16x16xf16", output)

        # Helper returns (row, col) = (4*(lane_id//16), lane_id%16)
        # A swaps to (col, row), then applies XOR swizzle
        # XOR swizzle: swizzled_col = (col_high XOR row_group) * 4 + col_low
        # Expected: (row, swizzled_col) pairs for 64 threads
        # Grouped by helper_row (which becomes col after swap):
        #   lane_id 0-15:   helper_row=0  -> col=0,  rows 0-15, swizzled_cols: 0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12
        #   lane_id 16-31:  helper_row=4  -> col=4,  rows 0-15, swizzled_cols: 4,4,4,4,0,0,0,0,12,12,12,12,8,8,8,8
        #   lane_id 32-47:  helper_row=8  -> col=8,  rows 0-15, swizzled_cols: 8,8,8,8,12,12,12,12,0,0,0,0,4,4,4,4
        #   lane_id 48-63:  helper_row=12 -> col=12, rows 0-15, swizzled_cols: 12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0
        expected = np.array([
            # lane_id 0-15:   row=0-15, col=0  -> swizzled_col pattern: 0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12
            0, 0, 1, 0, 2, 0, 3, 0, 4, 4, 5, 4, 6, 4, 7, 4,
            8, 8, 9, 8, 10, 8, 11, 8, 12, 12, 13, 12, 14, 12, 15, 12,
            # lane_id 16-31:  row=0-15, col=4  -> swizzled_col pattern: 4,4,4,4,0,0,0,0,12,12,12,12,8,8,8,8
            0, 4, 1, 4, 2, 4, 3, 4, 4, 0, 5, 0, 6, 0, 7, 0,
            8, 12, 9, 12, 10, 12, 11, 12, 12, 8, 13, 8, 14, 8, 15, 8,
            # lane_id 32-47:  row=0-15, col=8  -> swizzled_col pattern: 8,8,8,8,12,12,12,12,0,0,0,0,4,4,4,4
            0, 8, 1, 8, 2, 8, 3, 8, 4, 12, 5, 12, 6, 12, 7, 12,
            8, 0, 9, 0, 10, 0, 11, 0, 12, 4, 13, 4, 14, 4, 15, 4,
            # lane_id 48-63:  row=0-15, col=12 -> swizzled_col pattern: 12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0
            0, 12, 1, 12, 2, 12, 3, 12, 4, 8, 5, 8, 6, 8, 7, 8,
            8, 4, 9, 4, 10, 4, 11, 4, 12, 0, 13, 0, 14, 0, 15, 0,
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestSwizzledMfmaIndexB16x16xf16:
    """Test @swizzled_mfma_index_B_16x16xf16 function."""

    def test_swizzled_mfma_index_B(self):
        """Swizzled MFMA indexing for B fragment (XOR swizzle)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_swizzled_mfma_index_B_16x16xf16", output)

        # Helper returns (row, col) = (4*(lane_id//16), lane_id%16)
        # B applies XOR swizzle directly
        # XOR swizzle: swizzled_col = (col_high XOR row_group) * 4 + col_low
        # Expected: (row, swizzled_col) pairs for 64 threads
        # Grouped by row (from helper):
        #   lane_id 0-15:   row=0,  col=0-15  -> swizzled_col: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        #   lane_id 16-31:  row=4,  col=0-15  -> swizzled_col: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
        #   lane_id 32-47:  row=8,  col=0-15  -> swizzled_col: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
        #   lane_id 48-63:  row=12, col=0-15  -> swizzled_col: 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
        expected = np.array([
            # lane_id 0-15:   row=0, swizzled_col=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15,
            # lane_id 16-31:  row=4, swizzled_col=4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
            4, 4, 4, 5, 4, 6, 4, 7, 4, 0, 4, 1, 4, 2, 4, 3,
            4, 12, 4, 13, 4, 14, 4, 15, 4, 8, 4, 9, 4, 10, 4, 11,
            # lane_id 32-47:  row=8, swizzled_col=8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
            8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15,
            8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7,
            # lane_id 48-63:  row=12, swizzled_col=12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
            12, 12, 12, 13, 12, 14, 12, 15, 12, 8, 12, 9, 12, 10, 12, 11,
            12, 4, 12, 5, 12, 6, 12, 7, 12, 0, 12, 1, 12, 2, 12, 3,
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestSwizzledMfmaIndexC16x16xf32:
    """Test @swizzled_mfma_index_C_16x16xf32 function."""

    def test_swizzled_mfma_index_C(self):
        """Swizzled MFMA indexing for C fragment (XOR swizzle)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_swizzled_mfma_index_C_16x16xf32", output)

        # Helper returns (row, col) = (4*(lane_id//16), lane_id%16)
        # C applies XOR swizzle directly (same as B)
        # XOR swizzle: swizzled_col = (col_high XOR row_group) * 4 + col_low
        # Expected: (row, swizzled_col) pairs for 64 threads
        # Grouped by row (from helper):
        #   lane_id 0-15:   row=0,  col=0-15  -> swizzled_col: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        #   lane_id 16-31:  row=4,  col=0-15  -> swizzled_col: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
        #   lane_id 32-47:  row=8,  col=0-15  -> swizzled_col: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
        #   lane_id 48-63:  row=12, col=0-15  -> swizzled_col: 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
        expected = np.array([
            # lane_id 0-15:   row=0, swizzled_col=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15,
            # lane_id 16-31:  row=4, swizzled_col=4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
            4, 4, 4, 5, 4, 6, 4, 7, 4, 0, 4, 1, 4, 2, 4, 3,
            4, 12, 4, 13, 4, 14, 4, 15, 4, 8, 4, 9, 4, 10, 4, 11,
            # lane_id 32-47:  row=8, swizzled_col=8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
            8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15,
            8, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7,
            # lane_id 48-63:  row=12, swizzled_col=12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
            12, 12, 12, 13, 12, 14, 12, 15, 12, 8, 12, 9, 12, 10, 12, 11,
            12, 4, 12, 5, 12, 6, 12, 7, 12, 0, 12, 1, 12, 2, 12, 3,
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestIndexBxMxNxK:
    """Test @index_bxmxnxk_16x16x16_f16f16f32 function."""

    def test_index_bxmxnxk(self):
        """MFMA-style tiled indexing."""
        num_threads = 64
        output = np.zeros(num_threads, dtype=np.int32)
        compile_and_run("test_index_bxmxnxk", output)

        # Formula from indexing.mlir:
        # offset = bidx * num_waves * szI * szJ * tile_sz +
        #          widx * szI * szJ * tile_sz +
        #          i * szJ * tile_sz +
        #          j * tile_sz +
        #          lidx * lane_stride
        # With bidx=0, widx=0, i=0, j=0, szI=2, szJ=2, tile_sz=16, lane_stride=4, bdimx=64
        # num_waves = bdimx / 64 = 1
        # widx = tidx / 64 = 0
        # lidx = tidx % 64
        # offset = 0 + 0 + 0 + 0 + lidx * 4 = tidx * 4 (for tidx < 64)
        expected = np.array([
            0, 4, 8, 12, 16, 20, 24, 28,  # tid 0-7
            32, 36, 40, 44, 48, 52, 56, 60,  # tid 8-15
            64, 68, 72, 76, 80, 84, 88, 92,  # tid 16-23
            96, 100, 104, 108, 112, 116, 120, 124,  # tid 24-31
            128, 132, 136, 140, 144, 148, 152, 156,  # tid 32-39
            160, 164, 168, 172, 176, 180, 184, 188,  # tid 40-47
            192, 196, 200, 204, 208, 212, 216, 220,  # tid 48-55
            224, 228, 232, 236, 240, 244, 248, 252,  # tid 56-63
        ], dtype=np.int32)

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    TestGridPartition2D().test_grid_partition_2x4()
