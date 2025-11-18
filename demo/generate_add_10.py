#!/usr/bin/env python3
"""Demo: construct add_10 kernel IR programmatically using the Python API."""

import argparse
import os
import sys

# Add project root to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aster import ir
from aster.dialects import amdgcn, arith, builtin

KERNEL_NAME = "kernel"


def build_add_10_module(ctx: ir.Context, num_add_instructions: int) -> builtin.ModuleOp:
    """Build the add_10 kernel module programmatically."""
    module = builtin.ModuleOp()

    with ir.InsertionPoint(module.body):
        # Create amdgcn.module with target gfx942 and CDNA3 ISA
        amdgcn_mod = amdgcn.ModuleOp(
            amdgcn.Target.GFX942, amdgcn.ISAVersion.CDNA3, "add_10_module"
        )
        amdgcn_mod.body_region.blocks.append()

        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            # Create the kernel
            kernel = amdgcn.KernelOp(KERNEL_NAME)
            kernel.body_region.blocks.append()

            with ir.InsertionPoint(kernel.body_region.blocks[0]):
                # Allocate VGPRs with specific register numbers
                res, lhs, rhs = [amdgcn.api.alloca_vgpr(reg=i) for i in range(10, 13)]

                # Initialize registers with constants using v_mov_b32_e32
                int_type = ir.IntegerType.get_signless(32, ctx)
                lhs = amdgcn.api.v_mov_b32_e32(lhs, arith.constant(int_type, 1))
                rhs = amdgcn.api.v_mov_b32_e32(rhs, arith.constant(int_type, 2))

                # Perform sequential adds using VOP2 v_add_u32, DPS style.
                res = amdgcn.api.v_add_u32(res, lhs, rhs)
                for _ in range(num_add_instructions - 1):
                    res = amdgcn.api.v_add_u32(res, res, rhs)

                # If needed, could do a check of the value against an expected
                # value and trap. See e.g. test/python/cdna/test_cdna.py

                amdgcn.EndKernelOp()

    module.verify()
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
