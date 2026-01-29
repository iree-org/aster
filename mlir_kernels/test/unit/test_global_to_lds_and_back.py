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
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_mlir_file(file_name: str):
    """Get path to the test MLIR file."""
    return os.path.join(os.path.dirname(__file__), file_name)


def compile_and_run(
    file_name: str,
    kernel_name: str,
    input_data: list,
    output_data: np.ndarray,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
):
    """Compile and run a test kernel, returning the output buffer."""
    mlir_file = get_mlir_file(file_name)
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


class TestGlobalToLdsAndBack16x16:
    """Test @global_to_lds_and_back_wave_16x16xf16_wait function."""

    def test_copy_tile_at_position_3_5(self):
        """Copy a single 16x16 tile from position (3,5) in a 64x96 array.

        This tests that position handling is correct by loading from a specific tile
        location and verifying the correct data is retrieved.
        """
        rows, cols = 40, 60
        tile_m, tile_n = 1, 2  # Tile position (element 16, 32)
        m_pos, n_pos = tile_m * 16, tile_n * 16

        # Input: rows x cols matrix with values = linear index
        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run(
            "test_global_to_lds_and_back.mlir",
            "test_global_to_lds_and_back_wave_16x16xf16_wait",
            [input_data],
            output,
        )

        # Expected: the 16x16 tile at (48, 80) flattened
        input_2d = input_data.reshape(rows, cols)
        expected = input_2d[m_pos : m_pos + 16, n_pos : n_pos + 16]
        output_2d = output.reshape(rows, cols)
        output_2d_slice = output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16]
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output_2d_slice, expected)

            # remove output_2d_slice from output_2d at m_pos:m_pos+16, n_pos:n_pos+16
            # and assert all zeros.
            output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16] = (
                output_2d[m_pos : m_pos + 16, n_pos : n_pos + 16] - output_2d_slice
            )
            np.testing.assert_array_equal(output_2d.flatten(), np.zeros(rows * cols))


if __name__ == "__main__":
    # Run all tests
    TestGlobalToLdsAndBack16x16().test_copy_tile_at_position_3_5()
