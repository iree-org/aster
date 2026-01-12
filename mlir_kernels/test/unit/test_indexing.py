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
        expected = np.zeros(num_threads, dtype=np.int32)
        for tid in range(64):
            i = tid // 8
            j = tid % 8
            expected[tid] = (i * 16 + j) * 4

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
        expected = np.zeros(num_threads, dtype=np.int32)
        for tid in range(64):
            ii = tid // 8
            jj = tid % 8
            expected[tid] = ((0 + ii) * 16 + (0 + jj)) * 4

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
        expected = np.zeros(num_threads, dtype=np.int32)
        for tid in range(64):
            iii = tid // 8
            jjj = tid % 8
            expected[tid] = ((0 + 0 + iii) * 16 + (0 + 0 + jjj)) * 4

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndex16x16Helper:
    """Test @mfma_index_16x16_helper function."""

    def test_mfma_index_16x16_helper(self):
        """Returns (4 * (lane_id / 16), lane_id mod 16)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_16x16_helper", output)

        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(64):
            lane_id = tid % 64
            expected[tid * 2] = 4 * (lane_id // 16)
            expected[tid * 2 + 1] = lane_id % 16

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
        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(64):
            lane_id = tid % 64
            helper_i = 4 * (lane_id // 16)
            helper_j = lane_id % 16
            # mfma_index_A returns (j, i) from helper, so (helper_j, helper_i)
            expected[tid * 2] = helper_j
            expected[tid * 2 + 1] = helper_i

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndexB16x16xf16:
    """Test @mfma_index_B_16x16xf16 function."""

    def test_mfma_index_B_16x16xf16(self):
        """MFMA indexing for B fragment (same as helper)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_B_16x16xf16", output)

        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(64):
            lane_id = tid % 64
            expected[tid * 2] = 4 * (lane_id // 16)
            expected[tid * 2 + 1] = lane_id % 16

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


class TestMfmaIndexC16x16xf32:
    """Test @mfma_index_C_16x16xf32 function."""

    def test_mfma_index_C_16x16xf32(self):
        """MFMA indexing for C fragment (same as helper)."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_mfma_index_C_16x16xf32", output)

        expected = np.zeros(num_threads * 2, dtype=np.int32)
        for tid in range(64):
            lane_id = tid % 64
            expected[tid * 2] = 4 * (lane_id // 16)
            expected[tid * 2 + 1] = lane_id % 16

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
        expected = np.zeros(num_threads, dtype=np.int32)
        for tid in range(64):
            expected[tid] = tid * 4

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    TestGridPartition2D().test_grid_partition_2x4()
