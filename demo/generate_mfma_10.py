#!/usr/bin/env python3
"""Demo: construct mfma_10 kernel IR programmatically using KernelBuilder."""

import argparse

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder


def build_mfma_10_module(ctx: ir.Context, num_mfma_instructions: int) -> ir.Module:
    """Build the mfma_10 kernel module programmatically."""
    b = KernelBuilder("mfma_10_module", "kernel", target="gfx942")

    for _ in range(num_mfma_instructions):
        a = b.alloc_vgprx2()
        b_reg = b.alloc_vgprx2()
        b.alloc_vgprx4()
        acc = b.init_agprx4(b.constant_i32(0))
        b.mfma("v_mfma_f32_16x16x16_f16", acc, a, b_reg)

    module = b.build()
    module.operation.verify()
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Generate MFMA kernel IR programmatically"
    )
    parser.add_argument(
        "--num-mfma-instructions",
        type=int,
        default=10,
        help="Number of MFMA instructions to generate (default: 10)",
    )
    args = parser.parse_args()

    with ir.Context() as ctx, ir.Location.unknown():
        module = build_mfma_10_module(ctx, args.num_mfma_instructions)
        print(module)


if __name__ == "__main__":
    main()
