"""Compilation utilities for AMDGCN examples."""

import argparse
from typing import Callable, Optional

from aster import ir, utils
from aster._mlir_libs._amdgcn import SGPRType
from aster.dialects import amdgcn, builtin
from aster.dialects._amdgcn_ops_gen import ModuleOp, KernelOp, EndKernelOp
from aster.dialects._amdgcn_enum_gen import Target, ISAVersion
from aster.dialects.api import alloca_agpr, alloca_vgpr, alloca_sgpr


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--mcpu",
        type=str,
        default="gfx1201",
        help="Target architecture (default: gfx1201)",
    )
    parser.add_argument(
        "--wavefront-size",
        type=int,
        default=64,
        help="Wavefront size (default: 32)",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="kernel",
        help="Kernel name (default: kernel)",
    )


def build_module(
    ctx: ir.Context,
    inject_fn: Callable[[ir.Context, list, list, list, int], None],
    kernel_name: str = "kernel",
    mcpu: str = "gfx1201",
    num_sgprs: int = 16,
    num_vgprs: int = 16,
    num_agprs: int = 0,
    num_iterations: int = 1,
    use_relocatable_registers: bool = False,
    reserved_sgprs: int = 4,
    reserved_vgprs: int = 4,
    reserved_agprs: int = 4,
) -> ModuleOp:
    """Build an AMDGCN module with a kernel.

    Args:
        ctx: MLIR context
        inject_fn: Function that injects operations into the kernel. Should accept (ctx, sgprs, vgprs, agprs, num_iterations).
        kernel_name: Name of the kernel
        mcpu: Target architecture (gfx1201 or gfx942)
        num_sgprs: Number of SGPRs to allocate
        num_vgprs: Number of VGPRs to allocate
        num_agprs: Number of AGPRs to allocate (default: 0)
        num_iterations: Number of iterations for the inject function
        use_relocatable_registers: If True, allocate relocatable registers (default: False)
        reserved_sgprs: Number of reserved SGPRs to skip (default: 4)
        reserved_vgprs: Number of reserved VGPRs to skip (default: 4)
        reserved_agprs: Number of reserved AGPRs to skip (default: 4)

    Returns:
        AMDGCN ModuleOp
    """
    # Determine target and ISA version from mcpu
    if mcpu == "gfx942":
        target = Target.GFX942
        isa_version = ISAVersion.CDNA3
    elif mcpu == "gfx1201":
        target = Target.GFX1201
        isa_version = ISAVersion.RDNA4
    else:
        raise ValueError(f"Unsupported mcpu: {mcpu}")

    with ir.Location.unknown(ctx):
        # Create the module
        module = ModuleOp(
            target=target, isa_version=isa_version, sym_name=kernel_name + "_module"
        )
        module.body_region.blocks.append()

        with ir.InsertionPoint(module.body_region.blocks[0]):
            # Create kernel
            kernel = KernelOp(kernel_name)
            kernel.body_region.blocks.append()

            with ir.InsertionPoint(kernel.body_region.blocks[0]):
                # Allocate registers
                if use_relocatable_registers:
                    sgprs = [alloca_sgpr(reg=None).result for _ in range(num_sgprs)]
                    vgprs = [alloca_vgpr(reg=None).result for _ in range(num_vgprs)]
                    agprs = [alloca_agpr(reg=None).result for _ in range(num_agprs)]
                else:
                    sgprs = [alloca_sgpr(reg=i).result for i in range(num_sgprs)]
                    vgprs = [alloca_vgpr(reg=i).result for i in range(num_vgprs)]
                    agprs = [alloca_agpr(reg=i).result for i in range(num_agprs)]

                # Reserve first few registers, skip them before passing to inject_fn
                # This makes reserved registers transparent to downstream functions
                sgprs_available = (
                    sgprs[reserved_sgprs:] if reserved_sgprs > 0 else sgprs
                )
                vgprs_available = (
                    vgprs[reserved_vgprs:] if reserved_vgprs > 0 else vgprs
                )
                agprs_available = (
                    agprs[reserved_agprs:] if reserved_agprs > 0 else agprs
                )

                # Call inject function with available registers (reserved ones are skipped)
                inject_fn(
                    ctx,
                    sgprs_available,
                    vgprs_available,
                    agprs_available,
                    num_iterations,
                )

                # End kernel
                EndKernelOp()

    return module, kernel


def create_hsaco_binary(
    inject_fn: Callable[[ir.Context, list, list, list, int], None],
    output_path: str,
    kernel_name: str,
    mcpu: str,
    wavefront_size: int,
    *,
    num_sgprs: int = 16,
    num_vgprs: int = 16,
    num_agprs: int = 0,
    num_iterations: int = 1,
    dump_asm: bool = False,
    dump_ir: bool = False,
    use_relocatable_registers: bool = False,
    reserved_sgprs: int = 4,
    reserved_vgprs: int = 4,
    reserved_agprs: int = 4,
    kernel_args_fn: Optional[Callable[[], ir.Attribute]] = None,
):
    """Create HSACO binary from an inject function.

    Args:
        inject_fn: Function that injects operations
        output_path: Path to output .hsaco file
        kernel_name: Name of the kernel
        mcpu: Target architecture
        wavefront_size: Wavefront size (32 or 64)
        num_sgprs: Number of SGPRs
        num_vgprs: Number of VGPRs
        num_agprs: Number of AGPRs (default: 0)
        num_iterations: Number of iterations
        dump_asm: If True, print assembly
        dump_ir: If True, print IR
        use_relocatable_registers: If True, use relocatable registers (default: False)
        reserved_sgprs: Number of reserved SGPRs to skip (default: 4)
        reserved_vgprs: Number of reserved VGPRs to skip (default: 4)
        reserved_agprs: Number of reserved AGPRs to skip (default: 4)
    """
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True

        module, kernel = build_module(
            ctx,
            inject_fn,
            kernel_name,
            mcpu,
            num_sgprs,
            num_vgprs,
            num_agprs,
            num_iterations,
            use_relocatable_registers,
            reserved_sgprs,
            reserved_vgprs,
            reserved_agprs,
        )
        if kernel_args_fn is not None:
            kernel.arguments = kernel_args_fn()

        kernel.wavefront_size32 = wavefront_size == 32

        # Apply CSE pass before translation
        from aster._mlir_libs._mlir import passmanager

        # Create pass manager with CSE pass
        # Apply CSE at both module and function level
        pm = passmanager.PassManager.parse("amdgcn.module(amdgcn.kernel(cse))", ctx)
        pm.run(module)

        if dump_ir:
            print(module)

        # If we use relocatable registers, do not go to asm for now, we need explicit register allocation
        if use_relocatable_registers:
            print("\nNote: Skipping assembly generation with relocatable registers.")
            print(
                "Relocatable registers require explicit register allocation pass before ASM generation."
            )
            return None

        # Translate to assembly
        asm = utils.translate_module(module, debug_print=False)

        if dump_asm:
            print(asm)

        # Assemble to HSACO
        hsaco_path = utils.assemble_to_hsaco(
            asm,
            target=mcpu,
            wavefront_size=wavefront_size,
            output_path=output_path,
        )
        print(f"Kernel HSACO saved to {output_path}")

        return hsaco_path


def main(
    inject_fn: Callable[[ir.Context, list, list, list, int], None],
    *,
    kernel_name: str | None = None,
    kernel_args_fn: Optional[Callable[[], ir.Attribute]] = None,
):
    """Main entry point for kernel compilation.

    Args:
        inject_fn: Function that injects operations
        kernel_name: Name of the kernel
        kernel_args: Optional list of kernel argument metadata dicts
    """
    parser = argparse.ArgumentParser(description="AMDGCN kernel compiler")
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output .hsaco file path"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--num-sgprs",
        type=int,
        default=16,
        help="Number of SGPRs (default: 16)",
    )
    parser.add_argument(
        "--num-vgprs",
        type=int,
        default=16,
        help="Number of VGPRs (default: 16)",
    )
    parser.add_argument(
        "--num-agprs",
        type=int,
        default=0,
        help="Number of AGPRs (default: 0)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of iterations (default: 1)",
    )
    parser.add_argument(
        "--dump-asm",
        action="store_true",
        help="Dump assembly to stdout",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump IR to stdout",
    )
    parser.add_argument(
        "--use-relocatable-registers",
        action="store_true",
        help="Use relocatable (unallocated) registers instead of fixed register assignments",
    )
    parser.add_argument(
        "--reserved-sgprs",
        type=int,
        default=4,
        help="Number of reserved SGPRs to skip (default: 4)",
    )
    parser.add_argument(
        "--reserved-vgprs",
        type=int,
        default=4,
        help="Number of reserved VGPRs to skip (default: 4)",
    )
    parser.add_argument(
        "--reserved-agprs",
        type=int,
        default=4,
        help="Number of reserved AGPRs to skip (default: 4)",
    )

    args = parser.parse_args()

    # Use provided kernel_name or args.kernel_name
    final_kernel_name = kernel_name if kernel_name is not None else args.kernel_name

    # Compile kernel to HSACO
    print(f"Assembling kernel to {args.output}...")
    create_hsaco_binary(
        inject_fn,
        args.output,
        final_kernel_name,
        args.mcpu,
        args.wavefront_size,
        num_sgprs=args.num_sgprs,
        num_vgprs=args.num_vgprs,
        num_agprs=args.num_agprs,
        num_iterations=args.num_iterations,
        dump_asm=args.dump_asm,
        dump_ir=args.dump_ir,
        use_relocatable_registers=args.use_relocatable_registers,
        reserved_sgprs=args.reserved_sgprs,
        reserved_vgprs=args.reserved_vgprs,
        reserved_agprs=args.reserved_agprs,
        kernel_args_fn=kernel_args_fn,
    )
    print(f"Kernel HSACO saved to {args.output}")
