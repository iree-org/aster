"""CLI handling for CDNA3 matmul examples."""

import argparse
import sys
from typing import Callable

import numpy as np


def add_matmul_cli_args(
    parser: argparse.ArgumentParser, *, include_reserved: bool = True
) -> None:
    """Add matmul-specific CLI arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to
        include_reserved: If True, include reserved register arguments (default: True)
    """
    parser.add_argument(
        "--operand-register-size",
        type=int,
        default=2,
        help="Register range size for A and B operands (default: 2 for v_mfma_f32_16x16x16_f16)",
    )
    parser.add_argument(
        "--accum-register-size",
        type=int,
        default=4,
        help="Register range size for C accumulator (default: 4 for v_mfma_f32_16x16x16_f16)",
    )
    parser.add_argument(
        "--m-regs",
        type=int,
        default=1,
        help="Matrix dimension M registers (default: 1)",
    )
    parser.add_argument(
        "--n-regs",
        type=int,
        default=2,
        help="Matrix dimension N registers (default: 2)",
    )
    parser.add_argument(
        "--k-regs",
        type=int,
        default=16,
        help="Matrix dimension K registers (default: 16)",
    )
    if include_reserved:
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


def create_matmul_parser() -> argparse.ArgumentParser:
    """Create argument parser for matmul examples.

    Returns:
        ArgumentParser with matmul-specific arguments
    """
    parser = argparse.ArgumentParser(
        description="Matrix multiplication kernel with variant selection"
    )
    add_matmul_cli_args(parser)
    return parser


def create_matmul_kernel_args():
    """Create kernel arguments metadata for matmul kernel.

    Returns:
        The kernel arguments for A, B, C buffers
    """
    from aster.dialects import amdgcn

    kernel_args = [
        amdgcn.get_buffer_argument(
            access=amdgcn.AccessKind.ReadOnly,
            name="a",
        ),
        amdgcn.get_buffer_argument(
            access=amdgcn.AccessKind.ReadOnly,
            name="b",
        ),
        amdgcn.get_buffer_argument(
            access=amdgcn.AccessKind.ReadWrite,
            name="c",
        ),
    ]
    return amdgcn.get_kernel_arguments(kernel_args)


def compute_matmul_dimensions(
    m_regs: int,
    n_regs: int,
    k_regs: int,
    operand_register_size: int,
    accum_register_size: int,
) -> tuple[int, int, int]:
    """Compute matrix dimensions in elements from register configuration.

    Args:
        m_regs: Number of M dimension registers
        n_regs: Number of N dimension registers
        k_regs: Number of K dimension registers
        operand_register_size: Register range size for A and B operands
        accum_register_size: Register range size for C accumulator

    Returns:
        Tuple of (m_elements, n_elements, k_elements)
    """
    # MFMA op is 16x16x16, so each register range covers 16 elements
    m_elements = m_regs * accum_register_size * 16
    n_elements = n_regs * accum_register_size * 16
    k_elements = k_regs * operand_register_size * 16
    return m_elements, n_elements, k_elements


def create_matmul_arrays(
    m_regs: int,
    n_regs: int,
    k_regs: int,
    operand_register_size: int,
    accum_register_size: int,
    *,
    seed: int | None = None,
    dtype: type = np.float16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create numpy arrays for matmul kernel execution.

    Args:
        m_regs: Number of M dimension registers
        n_regs: Number of N dimension registers
        k_regs: Number of K dimension registers
        operand_register_size: Register range size for A and B operands
        accum_register_size: Register range size for C accumulator
        seed: Optional random seed for data generation
        dtype: NumPy dtype for arrays (default: np.float16)

    Returns:
        Tuple of (A, B, C) numpy arrays with proper dimensions
    """
    if seed is not None:
        np.random.seed(seed)

    m_elements, n_elements, k_elements = compute_matmul_dimensions(
        m_regs, n_regs, k_regs, operand_register_size, accum_register_size
    )

    A = np.random.randn(m_elements, k_elements).astype(dtype)
    B = np.random.randn(k_elements, n_elements).astype(dtype)
    C = np.zeros((m_elements, n_elements), dtype=dtype)

    return A, B, C


def run_matmul_cli(
    inject_fn: Callable,
    kernel_name: str = "kernel",
    *,
    add_args: bool = False,
) -> None:
    """Run matmul kernel generation with CLI argument parsing.

    Args:
        inject_fn: Injection function that accepts (ctx, sgprs, vgprs, agprs, num_iterations, **kwargs)
        kernel_name: Name of the kernel (default: "kernel")
        kernel_args: Optional list of kernel argument metadata dicts.
                     If None, uses create_matmul_kernel_args() to generate default args.
    """
    # Parse matmul-specific args
    parser = create_matmul_parser()
    args, remaining_args = parser.parse_known_args()

    # Print parsed matmul arguments
    print("=== Matmul Configuration ===")
    print(f"  operand_register_size: {args.operand_register_size}")
    print(f"  accum_register_size: {args.accum_register_size}")
    print(f"  m_regs: {args.m_regs}")
    print(f"  n_regs: {args.n_regs}")
    print(f"  k_regs: {args.k_regs}")
    print(f"  reserved_sgprs: {args.reserved_sgprs}")
    print(f"  reserved_vgprs: {args.reserved_vgprs}")
    print(f"  reserved_agprs: {args.reserved_agprs}")
    print("==============================\n")

    # Use default kernel args if not provided
    kernel_args_fn = None
    if add_args:
        kernel_args_fn = create_matmul_kernel_args

    # Create an injection function that includes the parameters
    def inject_matmul_variant(ctx, sgprs, vgprs, agprs, num_iterations=1):
        return inject_fn(
            ctx,
            sgprs,
            vgprs,
            agprs,
            num_iterations,
            operand_register_size=args.operand_register_size,
            accum_register_size=args.accum_register_size,
            m_regs=args.m_regs,
            n_regs=args.n_regs,
            k_regs=args.k_regs,
        )

    # Override sys.argv to pass remaining args plus reserved register args to main
    # Reserved registers are handled in utils.py build_module, not in inject_fn
    original_argv = sys.argv
    reserved_args = [
        "--reserved-sgprs",
        str(args.reserved_sgprs),
        "--reserved-vgprs",
        str(args.reserved_vgprs),
        "--reserved-agprs",
        str(args.reserved_agprs),
    ]
    sys.argv = [sys.argv[0]] + remaining_args + reserved_args

    try:
        from utils import main

        main(
            inject_matmul_variant,
            kernel_name=kernel_name,
            kernel_args_fn=kernel_args_fn,
        )
    finally:
        sys.argv = original_argv
