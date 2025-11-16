"""
Saturating the SCAL unit
========================

Example demonstrating v_mov_b32_e32 instruction with immediate operand using the 32-bit encoding.

python ex_01_cdna3_vmov_imm_32b.py --output test.hsaco --mcpu gfx1201 --num-iterations 5 --dump-asm

Note: Default kernel name is "kernel", override with --kernel-name <name>.
"""

from typing import Optional
from aster import ir
from aster.dialects import arith
from aster.dialects.api import alloca_vgpr, v_mov_b32_e32
from utils import main


def _inject_v_mov_32b(
    ctx: ir.Context,
    sgprs: list,
    vgprs: list,
    agprs: Optional[list] = None,
    num_iterations: int = 1,
) -> None:
    """Inject v_mov_b32_e32 operations with immediate values.

    Args:
        sgprs: List of SGPR values (unused in this example)
        vgprs: List of VGPR values to use as destinations
        num_iterations: Number of iterations to perform
        ctx: MLIR context
    """
    for i in range(num_iterations):
        # Create constant for immediate value
        int_type = ir.IntegerType.get_signless(32, ctx)
        const_value = arith.constant(int_type, i)

        # Get destination VGPR (cycle through available VGPRs)
        vdst = vgprs[i % len(vgprs)]

        v_mov_b32_e32(vdst, const_value)


if __name__ == "__main__":
    """
    To run this example:

        python ex_02_cdna3_vmov_imm_32b.py --output test.hsaco --mcpu gfx1201
    """

    main(_inject_v_mov_32b)
