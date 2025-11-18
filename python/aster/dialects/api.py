#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python syntactic sugar for AMDGCN operations."""

from typing import Optional, List, Union

from aster._mlir_libs._amdgcn import VGPRType, VGPRRangeType, SGPRRangeType
from .. import ir
from ._amdgcn_ops_gen import (
    AllocaOp,
    MakeRegisterRangeOp,
)


def alloca_vgpr(reg: Optional[int] = None) -> AllocaOp:
    """Allocate a VGPR register.

    Args:
        ctx: MLIR context
        reg: Register number (None for relocatable)

    Returns:
        AllocaOp that produces a VGPR register
    """
    from aster._mlir_libs._amdgcn import VGPRType

    ctx = ir.Context.current
    vgpr_type = VGPRType.get(ctx, reg)
    return AllocaOp(vgpr_type)


def alloca_sgpr(reg: Optional[int] = None) -> AllocaOp:
    """Allocate an SGPR register.

    Args:
        ctx: MLIR context
        reg: Register number (None for relocatable)

    Returns:
        AllocaOp that produces an SGPR register
    """
    from aster._mlir_libs._amdgcn import SGPRType

    ctx = ir.Context.current
    sgpr_type = SGPRType.get(ctx, reg)
    return AllocaOp(sgpr_type)


def alloca_agpr(reg: Optional[int] = None) -> AllocaOp:
    """Allocate an AGPR register.

    Args:
        ctx: MLIR context
        reg: Register number (None for relocatable)

    Returns:
        AllocaOp that produces an AGPR register
    """
    from aster._mlir_libs._amdgcn import AGPRType

    ctx = ir.Context.current
    agpr_type = AGPRType.get(ctx, reg)
    return AllocaOp(agpr_type)


def make_register_range(inputs: List[ir.Value], *, results=None) -> MakeRegisterRangeOp:
    """Create a register range from a list of registers.

    Args:
        inputs: List of registers (can be AllocaOp or OpResult)
        results: Optional result type (auto-inferred if None)

    Returns:
        MakeRegisterRangeOp that produces a register range
    """
    # Handle both AllocaOp and OpResult
    input_values = [inp.result if hasattr(inp, "result") else inp for inp in inputs]
    return MakeRegisterRangeOp(inputs=input_values, results=results)


from ._amdgcn_inst_gen import *
