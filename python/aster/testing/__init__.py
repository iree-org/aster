"""Unified testing utilities for ASTER E2E tests.

This module consolidates compile-and-run infrastructure previously spread across:
- test/integration/test_utils.py (low-level: load, compile, execute, verify)
- mlir_kernels/test/unit/test_utils.py (mid-level: compile_and_run)

Usage:
    from aster.testing import compile_and_run

    compile_and_run(
        "test_my_kernel.mlir",
        "my_kernel",
        [input_data],
        output,
    )
"""

import os
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Callable, Optional, List, Any, Generator

from aster import utils
from aster.compiler import (
    load_mlir_module_from_file,
    compile_mlir_file_to_asm,
    compile_mlir_module_to_asm,
    PrintOptions,
)
from aster.utils.env import (
    MillisecondFormatter,
    aster_get_env_or_default,
    aster_get_logger,
    aster_log_info,
    aster_parse_bool_env,
    aster_should_log,
)
from aster.dialects import amdgcn
from aster._mlir_libs._runtime_module import (
    hip_init,
    hip_get_device_count,
    hip_set_device,
)
from aster.execution.hip import execute_hsaco
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE
from aster.test_pass_pipelines import TEST_SYNCHRONOUS_PASS_PIPELINE

# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------


@contextmanager
def hsaco_file(path: str) -> Generator[str, None, None]:
    """Context manager that cleans up an HSACO file on exit.

    Args:
        path: Path to the HSACO file

    Yields:
        The path to the HSACO file

    Example:
        hsaco_path = utils.assemble_to_hsaco(asm, target=mcpu)
        with hsaco_file(hsaco_path):
            execute_kernel_and_verify(hsaco_path=hsaco_path, ...)
    """
    try:
        yield path
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def execute_kernel_and_verify(
    hsaco_path: Optional[str],
    kernel_name: str,
    input_args: List[np.ndarray],
    output_args: List[np.ndarray],
    mcpu: str,
    wavefront_size: int = 32,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    verify_fn: Optional[Callable[[List[np.ndarray], List[np.ndarray]], None]] = None,
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
) -> List[int]:
    """Execute a GPU kernel and verify its results.

    Args:
        hsaco_path: Path to the HSACO file
        kernel_name: Name of the kernel function
        input_args: List of input numpy arrays
        output_args: List of output numpy arrays (will be modified in-place)
        mcpu: Target GPU architecture (e.g., "gfx942", "gfx1201")
        wavefront_size: Wavefront size (default: 32)
        grid_dim: Grid dimensions (default: (1, 1, 1))
        block_dim: Block dimensions (default: (64, 1, 1))
        verify_fn: Custom verification function that takes (input_args, output_args).
                   Only called on first iteration if provided. (default: None)
        num_iterations: Number of times to execute the kernel (default: 1)
        device_id: GPU device ID to use. If None, uses current device. (default: None)
        flush_llc: Optional FlushLLC instance to flush the LLC before each iteration.
                   Called outside of timing start/stop. Cleanup is handled automatically. (default: None)

    Returns:
        List of execution times in nanoseconds, one per iteration
    """

    def _is_scalar(arg):
        return isinstance(arg, (int, float, np.integer, np.floating))

    assert all(
        (_is_scalar(arg) or arg.size > 0) for arg in input_args + output_args
    ), "All NP arrays must have > 0 elements"

    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")

    times_ns = execute_hsaco(
        hsaco_path,
        kernel_name,
        input_args,
        output_args,
        grid_dim,
        block_dim,
        num_iterations,
        device_id=device_id,
        flush_llc=flush_llc,
    )
    if verify_fn is not None:
        verify_fn(input_args, output_args)
    return times_ns


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def make_grid_block_preprocess(grid_dim, block_dim):
    """Create a preprocess function that substitutes NUM_THREADS and NUM_BLOCKS."""

    def preprocess(x):
        num_threads = block_dim[0] * block_dim[1] * block_dim[2]
        num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
        x = x.replace("{{NUM_THREADS}}", str(num_threads))
        x = x.replace("{{NUM_BLOCKS}}", str(num_blocks))
        return x

    return preprocess


# ---------------------------------------------------------------------------
# High-level compile-and-run
# ---------------------------------------------------------------------------

# Default test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


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
    verify_fn: Optional[Callable] = None,
    mcpu: str = MCPU,
    wavefront_size: int = WAVEFRONT_SIZE,
    num_iterations: int = 1,
    print_timings: bool = False,
    print_preprocessed_ir: bool = False,
    mlir_dir: Optional[str] = None,
    ctx=None,
):
    """Compile and run a test kernel, returning the output buffer.

    This is the unified entry point for "give me MLIR, run it on GPU, give me
    results". It handles context creation, compilation, assembly, GPU skip
    detection, and kernel execution.

    Args:
        file_name: Name of the MLIR file. If not an absolute path, resolved
            relative to mlir_dir (which defaults to the caller's directory).
        kernel_name: Name of the kernel to compile and run.
        input_data: Input numpy array(s). Single array or list of arrays.
        output_data: Output numpy array(s). Single array or list of arrays.
        grid_dim: Grid dimensions for kernel launch.
        block_dim: Block dimensions for kernel launch.
        library_paths: Library paths for preload. If None, auto-detected.
        preprocess: Optional preprocessing function for MLIR content.
        print_ir_after_all: Whether to print IR after all passes.
        pass_pipeline: Pass pipeline string. Defaults to TEST_SYNCHRONOUS_PASS_PIPELINE.
        verify_fn: Optional verification function(input_args, output_args).
        mcpu: Target GPU architecture (default: "gfx942").
        wavefront_size: Wavefront size (default: 64).
        num_iterations: Number of kernel executions (default: 1).
        print_timings: Whether to print pass timings.
        print_preprocessed_ir: Whether to print preprocessed IR.
        mlir_dir: Directory to resolve relative file_name against. Defaults to
            the caller's directory (via inspect.stack).
        ctx: MLIR context. If None, one is created internally.

    Returns:
        List of iteration times in nanoseconds, or None if skipped.
    """
    import pytest

    # Resolve MLIR file path
    if os.path.isabs(file_name):
        mlir_file = file_name
    else:
        if mlir_dir is None:
            import inspect

            caller_frame = inspect.stack()[1]
            mlir_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
        mlir_file = os.path.join(mlir_dir, file_name)

    # Auto-detect library paths if not provided
    if library_paths is None:
        try:
            from mlir_kernels.common import get_library_paths

            library_paths = get_library_paths()
        except ImportError:
            library_paths = []

    if pass_pipeline is None:
        pass_pipeline = TEST_SYNCHRONOUS_PASS_PIPELINE

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

    from aster import ir

    owns_ctx = ctx is None
    if owns_ctx:
        ctx = ir.Context()
        ctx.__enter__()

    try:
        asm_complete, module_after_passes = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            library_paths=library_paths,
            preprocess=preprocess,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=print_ir_after_all,
                print_timings=print_timings,
                print_preprocessed_ir=print_preprocessed_ir,
            ),
        )

        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=mcpu, wavefront_size=wavefront_size
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=mcpu):
                pytest.skip(
                    f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
                )

            iteration_times = execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=input_data,
                output_args=output_data,
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                grid_dim=grid_dim,
                block_dim=block_dim,
                verify_fn=verify_fn,
                num_iterations=num_iterations,
            )

            if num_iterations > 1:
                print(f"Iteration times: {iteration_times} nanoseconds")

            return iteration_times
    finally:
        if owns_ctx:
            ctx.__exit__(None, None, None)
