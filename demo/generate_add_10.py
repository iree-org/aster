#!/usr/bin/env python3
"""Demo: construct add_10 kernel IR programmatically using KernelBuilder."""

import argparse

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder


def build_add_10_module(ctx: ir.Context, num_add_instructions: int) -> ir.Module:
    """Build the add_10 kernel module programmatically."""
    b = KernelBuilder("add_10_module", "kernel", target="gfx942")

    # Allocate VGPRs
    res = b.alloca_vgpr()
    lhs = b.alloca_vgpr()
    rhs = b.alloca_vgpr()

    # Initialize with constants via v_add_u32 (lhs = 0 + 1, rhs = 0 + 2)
    c1 = b.constant_i32(1)
    c2 = b.constant_i32(2)
    b.vop2("v_add_u32", c1, lhs)
    b.vop2("v_add_u32", c2, rhs)

    # Perform sequential adds
    b.v_add_u32(lhs, rhs)
    for _ in range(num_add_instructions - 1):
        b.v_add_u32(res, rhs)

    module = b.build()
    module.operation.verify()
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Generate add kernel IR programmatically"
    )
    parser.add_argument(
        "--num-add-instructions",
        type=int,
        default=10,
        help="Number of add instructions to generate (default: 10)",
    )
    args = parser.parse_args()

    with ir.Context() as ctx, ir.Location.unknown():
        module = build_add_10_module(ctx, args.num_add_instructions)
        print(module)


if __name__ == "__main__":
    main()
