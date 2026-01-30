"""Common utilities for unit tests."""

import os
import pytest
import numpy as np
from typing import List, Optional, Callable

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)
from aster.pass_pipelines import TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_mlir_file(file_name: str) -> str:
    """Get path to a test MLIR file in the unit test directory."""
    return os.path.join(os.path.dirname(__file__), file_name)


def make_grid_block_preprocess(grid_dim, block_dim):
    """Create a preprocess function that substitutes NUM_THREADS and NUM_BLOCKS."""

    def preprocess(x):
        num_threads = block_dim[0] * block_dim[1] * block_dim[2]
        num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
        x = x.replace("{{NUM_THREADS}}", str(num_threads))
        x = x.replace("{{NUM_BLOCKS}}", str(num_blocks))
        return x

    return preprocess


def compile_and_run(
    file_name: str,
    kernel_name: str,
    input_data=None,
    output_data=None,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
    library_paths: Optional[List[str]] = None,
    preprocess: Optional[Callable[[str], str]] = None,
    print_ir_after_all: bool = False,
    pass_pipeline: Optional[str] = None,
):
    """Compile and run a test kernel, returning the output buffer.

    Args:
        file_name: Name of the MLIR file (relative to unit test directory)
        kernel_name: Name of the kernel to compile and run
        input_data: List of input numpy arrays (or single array, will be converted to list)
        output_data: List of output numpy arrays (or single array, will be converted to list)
        grid_dim: Grid dimensions for kernel launch
        block_dim: Block dimensions for kernel launch
        library_paths: Optional list of library paths. If None, uses get_library_paths()
        preprocess: Optional preprocessing function for MLIR content
        print_ir_after_all: Whether to print IR after all passes
        pass_pipeline: Optional pass pipeline string. If None, uses TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
    """
    mlir_file = get_mlir_file(file_name)

    if library_paths is None:
        library_paths = get_library_paths()

    if pass_pipeline is None:
        pass_pipeline = TEST_SYNCHRONOUS_SROA_PASS_PIPELINE

    # Convert single arrays to lists for compatibility
    if input_data is not None and not isinstance(input_data, list):
        input_data = [input_data]
    if output_data is not None and not isinstance(output_data, list):
        output_data = [output_data]

    # Default to empty lists if not provided
    if input_data is None:
        input_data = []
    if output_data is None:
        output_data = []

    with ir.Context() as ctx:
        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            library_paths=library_paths,
            print_ir_after_all=print_ir_after_all,
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
                input_args=input_data,
                output_args=output_data,
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
