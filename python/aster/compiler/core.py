"""MLIR compilation utilities: translation, assembly, and MLIR file I/O."""

import datetime
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from aster.core.target import Target


@dataclass
class PrintOptions:
    """Options that control diagnostic output emitted during compilation."""

    print_ir_after_all: bool = False
    print_preprocessed_ir: bool = False
    print_asm: bool = False
    print_root_dir: Optional[str] = None
    print_timings: bool = False

    @classmethod
    def from_flags(
        cls,
        print_ir_after_all: bool = False,
        print_preprocessed_ir: bool = False,
        print_asm: bool = False,
        print_root_dir: Optional[str] = None,
        print_timings: bool = False,
    ) -> "PrintOptions":
        """Construct PrintOptions, letting environment variables override the flags.

        The following environment variables take precedence over the corresponding
        argument when set:

        - ``ASTER_PRINT_IR_AFTER_ALL`` overrides *print_ir_after_all*.
        - ``ASTER_PRINT_ASM``          overrides *print_asm*.
        - ``ASTER_PRINT_DIR``          overrides *print_root_dir*.
        """
        from aster.utils.env import aster_get_env_or_default

        return cls(
            print_ir_after_all=aster_get_env_or_default(
                "ASTER_PRINT_IR_AFTER_ALL", print_ir_after_all
            ),
            print_preprocessed_ir=print_preprocessed_ir,
            print_asm=aster_get_env_or_default("ASTER_PRINT_ASM", print_asm),
            print_root_dir=aster_get_env_or_default("ASTER_PRINT_DIR", print_root_dir),
            print_timings=print_timings,
        )


def translate_module(module, debug_print=False):
    """Translate an AMDGCN module to assembly.

    Args:
        module: The AMDGCN module to translate.
        debug_print: If True, print debug comments for AllocaOp and MakeRegisterRangeOp.

    Returns:
        The assembly string representation of the module.
    """
    from aster._mlir_libs._amdgcn import translate_module as _translate_module

    return _translate_module(module.operation, debug_print)


def compile_to_hsaco(
    asm_content,
    target: Union[Target, str] = "gfx942",
    wavefront_size: int = 64,
) -> Optional[bytes]:
    """Compile AMDGPU assembly to hsaco binary.

    Args:
        asm_content: The assembly string to compile.
        target: A Target instance or architecture string (e.g. 'gfx942').
        wavefront_size: Wavefront size (32 or 64). Ignored when target is a Target.
    """
    from aster._mlir_libs._amdgcn import compile_asm as _compile_asm
    from aster.ir import Location, Context

    if isinstance(target, Target):
        mcpu = target.mcpu
        wf = target.wavefront_size
    else:
        mcpu = target
        wf = wavefront_size

    with Context() as ctx:
        return _compile_asm(
            Location.unknown(), asm_content, mcpu, f"+wavefrontsize{wf}"
        )


def assemble_to_hsaco(
    asm_content,
    target: Union[Target, str] = "gfx942",
    wavefront_size: int = 64,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Assemble AMDGPU assembly to an hsaco file.

    Args:
        asm_content: The assembly string to assemble.
        target: A Target instance or architecture string (e.g. 'gfx942').
        wavefront_size: Wavefront size (32 or 64). Ignored when target is a Target.
        output_path: Optional output path. If None, a temporary file is created.

    Returns:
        Path to the generated hsaco file.
    """
    hsaco_data = compile_to_hsaco(asm_content, target, wavefront_size)
    if hsaco_data is None:
        return None

    if output_path is None:
        hsaco_file = tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False)
        output_path = hsaco_file.name
        hsaco_file.close()

    with open(output_path, "wb") as f:
        f.write(hsaco_data)

    return output_path


def load_mlir_module_from_file(
    file_path: str,
    ctx,
    preprocess: Optional[Callable[[str], str]] = None,
    print_preprocessed_ir: bool = False,
):
    """Load MLIR module from file.

    Args:
        file_path: Path to MLIR file.
        ctx: MLIR context.
        preprocess: Optional function to preprocess the MLIR string before parsing.
        print_preprocessed_ir: If True, print the preprocessed IR before parsing.
    """
    from aster._mlir_libs._mlir import ir as mlir_ir

    with open(file_path, "r") as f:
        mlir_content = f.read()

    if preprocess is not None:
        mlir_content = preprocess(mlir_content)

    if print_preprocessed_ir:
        print(f"Preprocessed IR from {file_path}:\n{mlir_content}", flush=True)

    ctx.allow_unregistered_dialects = True

    with mlir_ir.Location.unknown():
        module = mlir_ir.Module.parse(mlir_content, context=ctx)
    return module


def _create_print_dir(kernel_name: str, print_root_dir: str) -> pathlib.Path:
    """Create a unique timestamped subdirectory for print output."""
    root = pathlib.Path(print_root_dir)
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"print_root_dir must be a directory, not a file: {print_root_dir}"
        )
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    prefix = f"{kernel_name}_{now.strftime('%d%m%y_%H%M%S')}_"
    path = pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    print(f"Printing results to directory: {path}", flush=True)
    return path


def _apply_preload_pass(module, library_paths: List[str], mlir_file: str, ctx, logger):
    """Apply the amdgcn-preload-library pass for the given library paths."""
    from aster._mlir_libs._mlir import passmanager
    from aster.utils.logging import aster_log_info

    for lib_path in library_paths:
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Library file not found: {lib_path}. MLIR file: {mlir_file}"
            )
    aster_log_info(logger, "[COMPILE] Pre-applying preload-library pass")
    paths_str = ",".join(library_paths)
    preload_pass = (
        f"builtin.module(amdgcn-preload-library{{library-paths={paths_str}}})"
    )
    pm = passmanager.PassManager.parse(preload_pass, ctx)
    pm.run(module.operation)


def _run_pass_pipeline(
    module,
    pass_pipeline: str,
    ctx,
    opts: PrintOptions,
    print_dir_path: Optional[pathlib.Path],
    logger,
):
    """Apply the pass pipeline with optional IR printing and timing."""
    from aster._mlir_libs._mlir import passmanager
    from aster.utils.logging import aster_log_info

    aster_log_info(logger, "[COMPILE] Applying pass pipeline")
    pm = passmanager.PassManager.parse(pass_pipeline, ctx)
    # Leave this here, it's useful for debugging.
    if opts.print_ir_after_all:
        if print_dir_path is not None:
            pm.enable_ir_printing(
                print_after_all=True, tree_printing_dir_path=str(print_dir_path)
            )
        else:
            pm.enable_ir_printing(print_after_all=True)
    if opts.print_timings:
        pm.enable_timing()
    pm.run(module.operation)
    aster_log_info(logger, "[COMPILE] Pass pipeline completed")


def _find_amdgcn_kernel(module, kernel_name: Optional[str]):
    """Return the amdgcn.ModuleOp containing kernel_name, or None if not found.

    When *kernel_name* is None, returns the first amdgcn.ModuleOp found.
    """
    from aster.dialects import amdgcn

    for op in module.body:
        if not isinstance(op, amdgcn.ModuleOp):
            continue
        if kernel_name is None:
            return op
        for kernel_op in op.body_region.blocks[0].operations:
            if (
                isinstance(kernel_op, amdgcn.KernelOp)
                and kernel_op.sym_name.value == kernel_name
            ):
                return op
    return None


def compile_mlir_module_to_asm(
    module,
    pass_pipeline: Optional[str] = None,
    library_paths: Optional[List[str]] = None,
    kernel_name: Optional[str] = None,
    print_opts: Optional[PrintOptions] = None,
) -> str:
    """Compile an already-built MLIR module to assembly.

    Runs the pass pipeline on *module*, finds the amdgcn.module, and
    translates to assembly. Use this when the module is built programmatically
    (e.g. via KernelBuilder) rather than loaded from a file.

    Args:
        module: An MLIR module in the AMDGCN dialect.
        pass_pipeline: Pass pipeline string. Defaults to TEST_SROA_PASS_PIPELINE.
        library_paths: Optional AMDGCN library files to preload before the pipeline.
        kernel_name: If provided, search for a specific kernel inside the module.
            If None, uses the first amdgcn.module found.
        print_opts: Print and diagnostic options. If None, defaults to
            PrintOptions.from_flags() which applies environment variable overrides.

    Returns:
        Assembly string.
    """
    from aster.utils.logging import aster_get_logger, aster_log_info

    if pass_pipeline is None:
        from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

        pass_pipeline = TEST_SROA_PASS_PIPELINE

    opts = print_opts if print_opts is not None else PrintOptions.from_flags()
    ctx = module.context
    logger = aster_get_logger()

    if library_paths:
        _apply_preload_pass(module, library_paths, "<module>", ctx, logger)

    _run_pass_pipeline(module, pass_pipeline, ctx, opts, None, logger)

    amdgcn_module = _find_amdgcn_kernel(module, kernel_name)
    assert amdgcn_module is not None, (
        f"failed to find kernel {kernel_name}"
        if kernel_name
        else "no amdgcn.module found after pipeline"
    )

    return translate_module(amdgcn_module, debug_print=False)


def compile_mlir_file_to_asm(
    mlir_file: str,
    kernel_name: str,
    pass_pipeline: str,
    ctx,
    preprocess: Optional[Callable[[str], str]] = None,
    library_paths: Optional[List[str]] = None,
    print_opts: Optional[PrintOptions] = None,
) -> Tuple[str, Any]:
    """Compile MLIR file to assembly and extract kernel name.

    Args:
        mlir_file: Path to MLIR file.
        kernel_name: Name of the kernel function.
        pass_pipeline: Pass pipeline string.
        ctx: MLIR context.
        preprocess: Optional function to preprocess the MLIR string before parsing.
        library_paths: Optional list of paths to AMDGCN library files to preload.
        print_opts: Print and diagnostic options. If None, defaults to
            PrintOptions.from_flags() which applies environment variable overrides.

    Returns:
        Tuple of (asm_code, module) where module is the MLIR module after passes.
    """
    from aster.utils.logging import aster_get_logger, aster_log_info

    opts = print_opts if print_opts is not None else PrintOptions.from_flags()

    print_dir_path: Optional[pathlib.Path] = None
    if opts.print_root_dir is not None and (opts.print_asm or opts.print_ir_after_all):
        print_dir_path = _create_print_dir(kernel_name, opts.print_root_dir)

    logger = aster_get_logger()
    aster_log_info(
        logger, f"[COMPILE] Loading MLIR file: {pathlib.Path(mlir_file).name}"
    )

    module = load_mlir_module_from_file(
        mlir_file, ctx, preprocess, opts.print_preprocessed_ir
    )

    if library_paths:
        _apply_preload_pass(module, library_paths, mlir_file, ctx, logger)

    if print_dir_path is not None:
        (print_dir_path / "pipeline.txt").write_text(pass_pipeline)
        (print_dir_path / "preprocessed.mlir").write_text(str(module))

    _run_pass_pipeline(module, pass_pipeline, ctx, opts, print_dir_path, logger)

    aster_log_info(logger, f"[COMPILE] Searching for kernel: {kernel_name}")
    amdgcn_module = _find_amdgcn_kernel(module, kernel_name)
    assert amdgcn_module is not None, f"failed to find kernel {kernel_name}"
    aster_log_info(logger, f"[COMPILE] Found kernel: {kernel_name}")

    aster_log_info(logger, "[COMPILE] Translating to assembly")
    asm_complete = translate_module(amdgcn_module, debug_print=False)
    aster_log_info(logger, "[COMPILE] Assembly generation completed")

    if opts.print_asm:
        if print_dir_path is not None:
            (print_dir_path / "kernel.s").write_text(asm_complete)
        else:
            print(asm_complete, flush=True)

    if print_dir_path is not None:
        (print_dir_path / "output.mlir").write_text(str(module))

    return asm_complete, module
