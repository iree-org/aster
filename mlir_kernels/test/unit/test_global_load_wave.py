"""Unit tests for global_load_wave library functions."""

import os
import pytest
import numpy as np
from typing import List

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


def get_mlir_file():
    """Get path to the test MLIR file."""
    return os.path.join(os.path.dirname(__file__), "test_global_load_wave.mlir")


def compile_and_run(
        kernel_name: str,
        input_data: np.ndarray,
        output_data: List[np.ndarray],
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
                output_args=output_data,
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


class TestGlobalLoadWave:
    """Test @global_load_wave_256xf16_via_dwordx2_wait function."""

    def test_global_load_ds_write(self):
        input = np.arange(1024, dtype=np.uint8)

        output_vx1 = np.zeros(256, dtype=np.uint8)
        output_vx2 = np.zeros(512, dtype=np.uint8)
        output_vx3 = np.zeros(768, dtype=np.uint8)
        output_vx4 = np.zeros(1024, dtype=np.uint8)

        compile_and_run("test_global_load_wave", [input], [output_vx1, output_vx2, output_vx3, output_vx4])

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output_vx1, input[:256])
            np.testing.assert_array_equal(output_vx2, input[:512])
            np.testing.assert_array_equal(output_vx3, input[:768])
            np.testing.assert_array_equal(output_vx4, input[:1024])


if __name__ == "__main__":
    # Run a specific test for debugging
    TestGlobalLoadWave().test_global_load_ds_write()
