#!/usr/bin/env python3
"""Demo: construct mfma_10 kernel IR programmatically using the Python API."""

import argparse
import os
import sys

# Add project root to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aster import ir
from aster.dialects import amdgcn, arith, builtin

KERNEL_NAME = "kernel"


def build_mfma_10_module(
    ctx: ir.Context, num_mfma_instructions: int
) -> builtin.ModuleOp:
    """Build the mfma_10 kernel module programmatically."""
    module = builtin.ModuleOp()

    with ir.InsertionPoint(module.body):
        # Create amdgcn.module with target gfx942 and CDNA3 ISA
        amdgcn_mod = amdgcn.ModuleOp(
            amdgcn.Target.GFX942, amdgcn.ISAVersion.CDNA3, "mfma_10_module"
        )
        amdgcn_mod.body_region.blocks.append()

        with ir.InsertionPoint(amdgcn_mod.body_region.blocks[0]):
            # Create the kernel
            kernel = amdgcn.KernelOp(KERNEL_NAME)
            kernel.body_region.blocks.append()

            with ir.InsertionPoint(kernel.body_region.blocks[0]):
                # Perform sequential MFMA operations using v_mfma_f32_16x16x16_f16
                # Each MFMA uses independent registers to avoid dependencies
                for i in range(num_mfma_instructions):
                    # Allocate registers for MFMA: A (2), B (2), C (4), D (4)
                    # Each iteration uses new registers starting at 8 + i * 12
                    base_reg = 8 + i * 12
                    a1, a2 = [
                        amdgcn.api.alloca_vgpr(base_reg + idx) for idx in range(2)
                    ]
                    b1, b2 = [
                        amdgcn.api.alloca_vgpr(base_reg + 2 + idx) for idx in range(2)
                    ]
                    c1, c2, c3, c4 = [
                        amdgcn.api.alloca_vgpr(base_reg + 4 + idx) for idx in range(4)
                    ]
                    # Allocate indepedent destination range for this MFMA
                    d1, d2, d3, d4 = [
                        amdgcn.api.alloca_vgpr(base_reg + 8 + idx) for idx in range(4)
                    ]
                    a_range = amdgcn.api.make_register_range([a1, a2])
                    b_range = amdgcn.api.make_register_range([b1, b2])
                    c_range = amdgcn.api.make_register_range([c1, c2, c3, c4])
                    d_range = amdgcn.api.make_register_range([d1, d2, d3, d4])

                    # MFMA: D = A * B + C (each operation is independent)
                    amdgcn.api.v_mfma_f32_16x16x16_f16(
                        d_range, a_range, b_range, c_range
                    )

                # If needed, could do a check of the value against an expected
                # value and trap. See e.g. test/python/cdna/test_cdna.py

                amdgcn.EndKernelOp()

    module.verify()
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
