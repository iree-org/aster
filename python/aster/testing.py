"""High-level test utilities for aster kernels.

This module provides the compile_and_run helper that combines compilation and execution.
For lower-level utilities, see:
- aster.logging: Logging utilities
- aster.compilation: MLIR compilation utilities
- aster.execution: Kernel execution utilities
"""

import numpy as np
from typing import Tuple, Callable, Optional, List

from aster import ir, utils
from aster.pass_pipelines import TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
from aster.compilation import compile_mlir_file_to_asm
from aster.execution import (
    DEFAULT_MCPU,
    DEFAULT_WAVEFRONT_SIZE,
    hsaco_file,
    execute_kernel_and_verify,
)

# Re-export for convenience
from aster.logging import (
    MillisecondFormatter,
    _get_logger,
    _should_log,
    _log_info,
    _log_with_device,
)
from aster.compilation import load_mlir_module_from_file

__all__ = [
    # Constants (from execution)
    "DEFAULT_MCPU",
    "DEFAULT_WAVEFRONT_SIZE",
    # Logging (from logging)
    "MillisecondFormatter",
    "_get_logger",
    "_should_log",
    "_log_info",
    "_log_with_device",
    # Compilation (from compilation)
    "load_mlir_module_from_file",
    "compile_mlir_file_to_asm",
    # Execution (from execution)
    "hsaco_file",
    "execute_kernel_and_verify",
    # High-level helper
    "compile_and_run",
]


def compile_and_run(
    mlir_file: str,
    kernel_name: str,
    input_data: Optional[List[np.ndarray]] = None,
    output_data: Optional[List[np.ndarray]] = None,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    library_paths: Optional[List[str]] = None,
    preprocess: Optional[Callable[[str], str]] = None,
    print_ir_after_all: bool = False,
    pass_pipeline: Optional[str] = None,
    mcpu: str = DEFAULT_MCPU,
    wavefront_size: int = DEFAULT_WAVEFRONT_SIZE,
) -> None:
    """Compile and run a kernel, handling GPU availability checks.

    This is a high-level helper that combines MLIR compilation and kernel execution.
    It handles:
    - Loading and compiling MLIR to assembly
    - Assembling to HSACO
    - Checking GPU availability (skips test if GPU not available)
    - Executing the kernel and copying results back

    Args:
        mlir_file: Absolute path to MLIR file
        kernel_name: Name of the kernel to compile and run
        input_data: List of input numpy arrays
        output_data: List of output numpy arrays (modified in-place)
        grid_dim: Grid dimensions for kernel launch
        block_dim: Block dimensions for kernel launch
        library_paths: Optional list of library paths for preload
        preprocess: Optional preprocessing function for MLIR content
        print_ir_after_all: Whether to print IR after all passes
        pass_pipeline: Pass pipeline string (defaults to TEST_SYNCHRONOUS_SROA_PASS_PIPELINE)
        mcpu: Target GPU (default: gfx942)
        wavefront_size: Wavefront size (default: 64)
    """
    import pytest

    if pass_pipeline is None:
        pass_pipeline = TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
    if input_data is None:
        input_data = []
    if output_data is None:
        output_data = []

    with ir.Context() as ctx:
        asm, _ = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            library_paths=library_paths or [],
            print_ir_after_all=print_ir_after_all,
            preprocess=preprocess,
        )

        hsaco_path = utils.assemble_to_hsaco(
            asm, target=mcpu, wavefront_size=wavefront_size
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=mcpu):
                print(asm)
                pytest.skip(f"GPU {mcpu} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=input_data,
                output_args=output_data,
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
