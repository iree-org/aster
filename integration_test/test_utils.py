"""Common utilities for integration tests."""

import os
import time
import logging
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Callable, Optional, List, Any, Generator, Union

from aster import utils
from aster.dialects import amdgcn
from aster._mlir_libs._runtime_module import (
    hip_init,
    hip_module_load_data,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_device_synchronize,
    hip_free,
    hip_malloc,
    hip_memcpy_host_to_device,
    hip_memcpy_device_to_host,
    hip_module_unload,
    hip_function_free,
    hip_get_device_count,
    hip_set_device,
    hip_get_device,
    hip_event_create,
    hip_event_destroy,
    hip_event_record,
    hip_event_synchronize,
    hip_event_elapsed_time,
)
from integration_test.flush_llc import FlushLLC


class MillisecondFormatter(logging.Formatter):
    """Formatter that includes milliseconds in timestamps."""

    def formatTime(self, record, datefmt=None):
        """Format time with millisecond precision."""
        ct = self.converter(record.created)
        msecs = int(record.msecs)
        return f"{ct.tm_hour:02d}:{ct.tm_min:02d}:{ct.tm_sec:02d}.{msecs:03d}"


def _get_logger():
    """Get logger configured for multiprocessing-safe logging."""
    logger = logging.getLogger("benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            MillisecondFormatter(
                fmt="%(asctime)s [PID:%(process)d] %(message)s",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _should_log():
    """Check if logging is enabled via ASTER_LOGGING environment variable."""
    return bool(os.getenv("ASTER_LOGGING"))


def _log_with_device(logger, device_id, message):
    """Log message with device_id."""
    if not _should_log():
        return
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    logger.info(f"[{device_str}] {message}")


def _log_info(logger, message):
    """Log info message if logging is enabled."""
    if not _should_log():
        return
    logger.info(message)


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


# Default pass pipeline for integration tests
DEFAULT_SROA_PASS_PIPELINE = (
    "builtin.module("
    "  aster-selective-inlining,"
    "  cse,canonicalize,symbol-dce,"
    "  amdgcn-instruction-scheduling-autoschedule,"
    "  aster-op-scheduling,"
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    # Note: SROA requires inlining of everything and canonicalization of GPU
    # quantities to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    # Try wait optimization early
    # Note: analysis does not support branches so full inlining is required.
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: SROA requires inlining of everything and canonicalization of GPU
    # quantities to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the pass happens correctly..
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: allocates special registers and does not work with function calls.
    # This is really needed before optimizing straight-line waits otherwise we
    # may miss some dependencies (e.g. s_load_dwordx2 does not yet exist).
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    # Note: this must have a aster-amdgcn-expand-md-ops run before to expose
    # s_load_dwordx2.
    # TODO: NORMAL FORMS or include in pass.
    # Note: going to lsir early will make memory dependency more conservative,
    # resulting in more waits during amdgcn-optimize-straight-line-waits.
    # TODO: NORMAL FORMS or include in pass.
    "  amdgcn-optimize-straight-line-waits,"
    #
    # Note: convert to lsir and AMDGCN after straight-line wait optimization.
    # Note: aster-to-int-arith contains lower-affine without linking in and
    # cargo-culting the whole conversion library.
    "  aster-to-int-arith,"
    "  aster-optimize-arith,"
    "  aster-amdgcn-set-abi,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  canonicalize,cse,"
    "  canonicalize,"
    "  aster-to-amdgcn,"
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the interference graph is built correctly...
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: this is really needed to lower away threadidx etc ops into alloc that
    # can be relocated
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    #
    # Note: needs to know about instructions and actual register number for
    # WAW dependencies.
    "  amdgcn-nop-insertion{conservative-extra-delays=0}"
    ")"
)

# SROA pass pipeline that runs synchronously, i.e. no wait optimization and extra
# NOP insertion. This is used for debugging races.
SYNCHRONOUS_SROA_PASS_PIPELINE = (
    "builtin.module("
    "  aster-selective-inlining,"
    "  cse,canonicalize,symbol-dce,"
    "  amdgcn-instruction-scheduling-autoschedule,"
    "  aster-op-scheduling,"
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    # Note: SROA requires inlining of everything to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    # Try wait optimization early
    # Note: analysis does not support branches so full inlining is required.
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  cse,canonicalize,symbol-dce,"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: SROA requires inlining of everything to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the pass happens correctly..
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: allocates special registers and does not work with function calls.
    # This is really needed before optimizing straight-line waits otherwise we
    # may miss some dependencies (e.g. s_load_dwordx2 does not yet exist).
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    # Note: this must have a aster-amdgcn-expand-md-ops run before to expose
    # s_load_dwordx2.
    # TODO: NORMAL FORMS or include in pass.
    # Note: going to LSIR early will make memory dependency more conservative,
    # resulting in more waits during amdgcn-optimize-straight-line-waits.
    # TODO: NORMAL FORMS or include in pass.
    # "  amdgcn-optimize-straight-line-waits,"
    #
    # Note: convert to LSIR and AMDGCN after straight-line wait optimization.
    # Note: aster-to-int-arith contains lower-affine without linking in and
    # cargo-culting the whole conversion library.
    "  aster-to-int-arith,"
    "  aster-optimize-arith,"
    "  aster-amdgcn-set-abi,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  canonicalize,cse,"
    "  canonicalize,"
    "  aster-to-amdgcn,"
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the interference graph is built correctly...
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: this is really needed to lower away threadidx etc ops into alloc that
    # can be relocated
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    #
    # Note: needs to know about instructions and actual register number for
    # WAW dependencies.
    "  amdgcn-nop-insertion{conservative-extra-delays=32}"
    ")"
)


def load_mlir_module_from_file(
    file_path: str, ctx, preprocess: Optional[Callable[[str], str]] = None
):
    """Load MLIR module from file.

    Args:
        file_path: Path to MLIR file
        ctx: MLIR context
        preprocess: Optional function to preprocess the MLIR string before parsing
    """
    from aster._mlir_libs._mlir import ir as mlir_ir

    with open(file_path, "r") as f:
        mlir_content = f.read()

    if preprocess is not None:
        mlir_content = preprocess(mlir_content)

    # Enable unregistered dialects to allow parsing MLIR with unregistered dialects
    ctx.allow_unregistered_dialects = True

    with mlir_ir.Location.unknown():
        module = mlir_ir.Module.parse(mlir_content, context=ctx)
    return module


def compile_mlir_file_to_asm(
    mlir_file: str,
    kernel_name: str,
    pass_pipeline: str,
    ctx,
    preprocess: Optional[Callable[[str], str]] = None,
    print_ir_after_all: bool = False,
    library_paths: Optional[List[str]] = None,
    print_timings: bool = False,
) -> Tuple[str, Any]:
    """Compile MLIR file to assembly and extract kernel name.

    Args:
        mlir_file: Path to MLIR file
        kernel_name: Name of the kernel function
        pass_pipeline: Pass pipeline string
        ctx: MLIR context
        preprocess: Optional function to preprocess the MLIR string before parsing
        print_ir_after_all: If True, print the IR after all passes have been applied
        library_paths: Optional list of paths to AMDGCN library files to preload

    Returns:
        Tuple of (asm_code, module) where module is the MLIR module after passes
    """
    logger = _get_logger()
    _log_info(logger, f"[COMPILE] Loading MLIR file: {os.path.basename(mlir_file)}")

    module = load_mlir_module_from_file(mlir_file, ctx, preprocess)

    # Apply passes
    from aster._mlir_libs._mlir import passmanager

    # Pre-apply preload-library pass if library paths are provided
    if library_paths:
        # Verify all library files exist
        for lib_path in library_paths:
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"Library file not found: {lib_path}. " f"MLIR file: {mlir_file}"
                )
        _log_info(logger, f"[COMPILE] Pre-applying preload-library pass")
        paths_str = ",".join(library_paths)
        preload_pass = (
            f"builtin.module(amdgcn-preload-library{{library-paths={paths_str}}})"
        )
        pm = passmanager.PassManager.parse(preload_pass, ctx)
        pm.run(module.operation)

    _log_info(logger, f"[COMPILE] Applying pass pipeline")
    pm = passmanager.PassManager.parse(pass_pipeline, ctx)
    # Leave this here, it's useful for debugging.
    if print_ir_after_all:
        pm.enable_ir_printing()
    if print_timings:
        pm.enable_timing()
    pm.run(module.operation)
    _log_info(logger, f"[COMPILE] Pass pipeline completed")

    # Find the amdgcn.kernel inside the proper amdgcn.module
    _log_info(logger, f"[COMPILE] Searching for kernel: {kernel_name}")
    amdgcn_module = None
    found_kernel = False
    for op in module.body:
        if not isinstance(op, amdgcn.ModuleOp):
            continue
        amdgcn_module = op
        for kernel_op in amdgcn_module.body_region.blocks[0].operations:
            if not isinstance(kernel_op, amdgcn.KernelOp):
                continue
            if kernel_op.sym_name.value == kernel_name:
                found_kernel = True
                break
        if found_kernel:
            break

    assert amdgcn_module is not None, "Failed to find any AMDGCN module"
    assert found_kernel, f"Failed to find kernel {kernel_name}"
    _log_info(logger, f"[COMPILE] Found kernel: {kernel_name}")

    _log_info(logger, f"[COMPILE] Translating to assembly")
    asm_complete = utils.translate_module(
        amdgcn_module,
        debug_print=False,
    )
    _log_info(logger, f"[COMPILE] Assembly generation completed")

    # print(asm_complete)
    return asm_complete, module


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
    padding_bytes: Optional[List[int]] = None,
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[FlushLLC] = None,
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
        padding_bytes: List of padding bytes per buffer, one for each buffer in input_args + output_args.
                       If None, no padding is applied. (default: None)
        num_iterations: Number of times to execute the kernel (default: 1)
        device_id: GPU device ID to use. If None, uses current device. (default: None)
        flush_llc: Optional FlushLLC instance to flush the LLC before each iteration.
                   Called outside of timing start/stop. Cleanup is handled automatically. (default: None)

    Returns:
        List of execution times in nanoseconds, one per iteration
    """

    assert all(
        array.size > 0 for array in input_args + output_args
    ), "All NP arrays must have > 0 elements"

    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")

    logger = _get_logger()

    # Initialize HIP runtime and set device before any GPU allocations
    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    gpu_ptrs: Optional[List[Any]] = None
    padded_buffers: Optional[List[Any]] = None
    has_padding = False
    module = None
    function = None
    start_event = None
    stop_event = None

    actual_device_id = device_id if device_id is not None else hip_get_device()

    try:
        _log_with_device(
            logger,
            actual_device_id,
            f"Starting execution: kernel={kernel_name}, iterations={num_iterations}",
        )

        # Load hsaco binary
        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()

        module = hip_module_load_data(hsaco_binary)
        function = hip_module_get_function(module, kernel_name.encode())
        _log_with_device(
            logger, actual_device_id, f"Loaded HSACO: {os.path.basename(hsaco_path)}"
        )

        all_arrays = input_args + output_args

        # Normalize padding_bytes: if None, create list of zeros
        if padding_bytes is None:
            padding_bytes = [0] * len(all_arrays)
        elif len(padding_bytes) != len(all_arrays):
            raise ValueError(
                f"padding_bytes must have {len(all_arrays)} elements (one per buffer), "
                f"got {len(padding_bytes)}"
            )

        # Check if any buffer needs padding
        has_padding = any(pb > 0 for pb in padding_bytes)

        if has_padding:
            # Use padded buffers
            padded_buffers = []
            ptr_values = []

            # Allocate padded buffers and copy input data
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} padded buffers"
            )
            for arr, pad_bytes in zip(all_arrays, padding_bytes):
                base_ptr, data_ptr, _ = utils.copy_array_to_gpu(arr, pad_bytes)
                padded_buffers.append(base_ptr)
                ptr_value = utils.unwrap_pointer_from_capsule(data_ptr)
                ptr_values.append(ptr_value)

            # Create kernel arguments from padded buffer pointers
            params_tuple = utils.create_kernel_args_capsule(ptr_values)
        else:
            # Use normal buffers
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} buffers"
            )
            params_tuple, gpu_ptrs = utils.create_kernel_args_capsule_from_numpy(
                *all_arrays, device_id=device_id
            )

        iteration_times_ns = []

        # Create events for GPU-level timing
        start_event = hip_event_create()
        stop_event = hip_event_create()

        _log_with_device(
            logger,
            actual_device_id,
            f"Launching kernel: grid={grid_dim}, block={block_dim}",
        )

        if flush_llc is not None:
            flush_llc.initialize()

        # Execute kernel multiple times for cache warming
        for iteration in range(num_iterations):
            # Flush LLC before each iteration (outside timing)
            if flush_llc is not None:
                flush_llc.flush_llc()

            # Record start event
            hip_event_record(start_event)

            # Launch kernel
            hip_module_launch_kernel(
                function,
                grid_dim[0],
                grid_dim[1],
                grid_dim[2],
                block_dim[0],
                block_dim[1],
                block_dim[2],
                params_tuple[0],
            )

            # Record stop event and synchronize
            hip_event_record(stop_event)
            hip_event_synchronize(stop_event)

            # Get elapsed time in milliseconds, convert to nanoseconds
            elapsed_ms = hip_event_elapsed_time(start_event, stop_event)
            elapsed_ns = int(elapsed_ms * 1_000_000)
            iteration_times_ns.append(elapsed_ns)

            _log_with_device(
                logger, actual_device_id, f"Iteration {iteration}: {elapsed_ms:.3f}ms"
            )

            # Verify results only on first iteration
            if iteration == 0:
                _log_with_device(logger, actual_device_id, "Verifying results")
                # Copy results back
                num_inputs = len(input_args)
                if has_padding:
                    # Copy from padded buffers
                    assert padded_buffers is not None
                    for i, output_arr in enumerate(output_args):
                        base_ptr = padded_buffers[num_inputs + i]
                        pad_bytes = padding_bytes[num_inputs + i]
                        utils.copy_from_gpu_buffer(base_ptr, output_arr, pad_bytes)
                else:
                    # Copy from normal buffers
                    assert gpu_ptrs is not None
                    for i, output_arr in enumerate(output_args):
                        output_ptr = gpu_ptrs[num_inputs + i]
                        capsule_output = utils.wrap_pointer_in_capsule(
                            output_arr.ctypes.data
                        )
                        hip_memcpy_device_to_host(
                            capsule_output, output_ptr, output_arr.nbytes
                        )

                if verify_fn is not None:
                    verify_fn(input_args, output_args)
                    _log_with_device(logger, actual_device_id, "Verification passed")

        # Free the GPU buffers
        avg_time_ms = sum(iteration_times_ns) / len(iteration_times_ns) / 1_000_000
        _log_with_device(
            logger,
            actual_device_id,
            f"Completed {num_iterations} iterations, avg={avg_time_ms:.3f}ms",
        )

        return iteration_times_ns

    finally:
        # Cleanup events
        if start_event is not None:
            hip_event_destroy(start_event)
        if stop_event is not None:
            hip_event_destroy(stop_event)
        # Cleanup buffers
        if padded_buffers is not None:
            for ptr in padded_buffers:
                hip_free(ptr)
        elif gpu_ptrs is not None:
            for ptr in gpu_ptrs:
                hip_free(ptr)
        # Cleanup flush buffer if FlushLLC instance was provided
        if flush_llc is not None:
            flush_llc.cleanup()
        if function is not None:
            hip_function_free(function)
        if module is not None:
            hip_module_unload(module)
