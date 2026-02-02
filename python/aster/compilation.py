"""MLIR compilation utilities for aster."""

import os
from typing import Tuple, Callable, Optional, List, Any

from aster import utils
from aster.dialects import amdgcn
from aster.logging import _get_logger, _log_info

__all__ = [
    "load_mlir_module_from_file",
    "compile_mlir_file_to_asm",
]


def load_mlir_module_from_file(
    file_path: str, ctx, preprocess: Optional[Callable[[str], str]] = None
):
    """Load MLIR module from file.

    Args:
        file_path: Path to MLIR file
        ctx: MLIR context
        preprocess: Optional function to preprocess the MLIR string before parsing

    Returns:
        Parsed MLIR module
    """
    from aster._mlir_libs._mlir import ir as mlir_ir

    with open(file_path, "r") as f:
        mlir_content = f.read()

    if preprocess is not None:
        mlir_content = preprocess(mlir_content)

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
    """Compile MLIR file to assembly.

    Args:
        mlir_file: Path to MLIR file
        kernel_name: Name of the kernel function
        pass_pipeline: Pass pipeline string
        ctx: MLIR context
        preprocess: Optional function to preprocess the MLIR string before parsing
        print_ir_after_all: If True, print the IR after all passes
        library_paths: Optional list of paths to AMDGCN library files to preload
        print_timings: If True, print pass timings

    Returns:
        Tuple of (asm_code, module)
    """
    logger = _get_logger()
    _log_info(logger, f"[COMPILE] Loading MLIR file: {os.path.basename(mlir_file)}")

    module = load_mlir_module_from_file(mlir_file, ctx, preprocess)

    from aster._mlir_libs._mlir import passmanager

    # Pre-apply preload-library pass if library paths are provided
    if library_paths:
        for lib_path in library_paths:
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"Library file not found: {lib_path}. MLIR file: {mlir_file}"
                )
        _log_info(logger, "[COMPILE] Pre-applying preload-library pass")
        paths_str = ",".join(library_paths)
        preload_pass = (
            f"builtin.module(amdgcn-preload-library{{library-paths={paths_str}}})"
        )
        pm = passmanager.PassManager.parse(preload_pass, ctx)
        pm.run(module.operation)

    _log_info(logger, "[COMPILE] Applying pass pipeline")
    pm = passmanager.PassManager.parse(pass_pipeline, ctx)
    if print_ir_after_all:
        pm.enable_ir_printing()
    if print_timings:
        pm.enable_timing()
    pm.run(module.operation)
    _log_info(logger, "[COMPILE] Pass pipeline completed")

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

    _log_info(logger, "[COMPILE] Translating to assembly")
    asm_complete = utils.translate_module(amdgcn_module, debug_print=False)
    _log_info(logger, "[COMPILE] Assembly generation completed")

    return asm_complete, module
