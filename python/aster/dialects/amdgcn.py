#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aster.dialects._amdgcn_ops_gen import *
from aster.dialects._ods_common import (
    get_default_loc_context as _ods_get_default_loc_context,
)
from aster.dialects._amdgcn_enum_gen import *
from aster.dialects._amdgcn_inst_gen import *
from aster.dialects._ods_common import _cext as _ods_cext

_ods_ir = _ods_cext.ir

from aster import ir
from aster._mlir_libs._amdgcn import (
    AGPRRangeType,
    AGPRType,
    SGPRRangeType,
    SGPRType,
    VGPRRangeType,
    VGPRType,
)

from typing import List, Optional, Union

from aster.ir import register_attribute_builder


def alloca_vgpr(reg: Optional[int] = None) -> ir.Value:
    """Allocate a VGPR register."""
    return AllocaOp(VGPRType.get(ir.Context.current, reg)).result


def alloca_sgpr(reg: Optional[int] = None) -> ir.Value:
    """Allocate an SGPR register."""
    return AllocaOp(SGPRType.get(ir.Context.current, reg)).result


def alloca_agpr(reg: Optional[int] = None) -> ir.Value:
    """Allocate an AGPR register."""
    return AllocaOp(AGPRType.get(ir.Context.current, reg)).result


def make_register_range(inputs: List[ir.Value], *, results=None) -> ir.Value:
    """Create a register range from a list of registers."""
    input_values = [inp.result if hasattr(inp, "result") else inp for inp in inputs]
    return MakeRegisterRangeOp(inputs=input_values, results=results).result


@register_attribute_builder("AMDGCN_InstAttr")
def _instattr(x, context):
    return _ods_ir.Attribute.parse(f"#amdgcn.inst<{str(x)}>", context=context)


@register_attribute_builder("KernelArgumentsAttr")
def _kernelargumentsattr(x, context):
    args = ", ".join(str(arg) for arg in x)
    return _ods_ir.Attribute.parse(
        f"#amdgcn.kernel_arguments<[{args}]>", context=context
    )


@register_attribute_builder("ByValueArgAttr")
def _byvalueargattr(size, alignment, name, type, context):
    alg_str = "" if alignment is None else f", alignment = {alignment}"
    name_str = "" if not name else f', name = "{name}"'
    type_str = "" if type is None else f", type = {type}"
    args = f"size = {size}{alg_str}{name_str}{type_str}"
    return _ods_ir.Attribute.parse(f"#amdgcn.by_val_arg<{args}>", context=context)


@register_attribute_builder("BufferArg")
def _bufferargattr(address_space, access, flags, name, type, context):
    name_str = "" if name is None else f', name = "{name}"'
    type_str = "" if type is None else f", type = {type}"
    args = f"address_space = {str(address_space)}, access = {str(access)}, flags = {str(flags)}{name_str}{type_str}"
    return _ods_ir.Attribute.parse(f"#amdgcn.buffer_arg<{args}>", context=context)


def get_by_value_argument(
    size: int,
    alignment: Optional[int] = None,
    name: str = "",
    type: Optional[str] = None,
    ctx=None,
):
    if ctx is None:
        ctx = _ods_ir.Context.current
    return _ods_ir.AttrBuilder.get("ByValueArgAttr")(size, alignment, name, type, ctx)


def get_buffer_argument(
    address_space: AddressSpaceKind = AddressSpaceKind.Global,
    access: AccessKind = AccessKind.ReadWrite,
    flags: KernelArgumentFlags = KernelArgumentFlags.None_,
    name: str = "",
    type: Optional[str] = None,
    ctx=None,
):
    if ctx is None:
        ctx = _ods_ir.Context.current
    return _ods_ir.AttrBuilder.get("BufferArg")(
        address_space, access, flags, name, type, ctx
    )


def get_kernel_arguments(arguments: list[_ods_ir.Attribute], ctx=None):
    if ctx is None:
        ctx = _ods_ir.Context.current
    return _ods_ir.AttrBuilder.get("KernelArgumentsAttr")(arguments, ctx)


def _create_inst_op(name, opcode, outs, ins, *, loc=None, ip=None):
    """Create a DPS instruction op via ir.Operation.create.

    The ODS-generated Python classes have _ODS_RESULT_SEGMENTS = [0, 0]
    because DPS results are optional (present only when the register has
    value semantics). The C++ custom parser computes resultSegmentSizes
    explicitly; we do the same here. All register types constructed from
    Python have value semantics, so every present output operand
    produces a result.
    """
    result_types = []
    result_segments = []
    operand_segments = []
    operands = []
    for o in outs:
        if o is not None:
            operands.append(o)
            operand_segments.append(1)
            result_types.append(o.type)
            result_segments.append(1)
        else:
            operand_segments.append(0)
            result_segments.append(0)
    for i in ins:
        if i is not None:
            operands.append(i)
            operand_segments.append(1)
        else:
            operand_segments.append(0)

    return _ods_ir.Operation.create(
        name,
        results=result_types,
        operands=operands,
        attributes={
            "opcode": (
                opcode
                if isinstance(opcode, _ods_ir.Attribute)
                else _ods_ir.AttrBuilder.get("AMDGCN_InstAttr")(
                    opcode, context=_ods_get_default_loc_context(loc)
                )
            ),
            "operandSegmentSizes": _ods_ir.DenseI32ArrayAttr.get(operand_segments),
            "resultSegmentSizes": _ods_ir.DenseI32ArrayAttr.get(result_segments),
        },
        loc=loc,
        ip=ip,
    )


def vop2(opcode, dest, src0, src1, *, dst1=None, src2=None, loc=None, ip=None):
    """Create amdgcn.vop2."""
    op = _create_inst_op(
        "amdgcn.vop2", opcode, outs=[dest, dst1], ins=[src0, src1, src2], loc=loc, ip=ip
    )
    return op.results[0]


def vop3(opcode, dest, src0, src1, *, dst1=None, src2=None, loc=None, ip=None):
    """Create amdgcn.vop3."""
    op = _create_inst_op(
        "amdgcn.vop3", opcode, outs=[dest, dst1], ins=[src0, src1, src2], loc=loc, ip=ip
    )
    return op.results[0]


def cmpi(opcode, dest, lhs, rhs, *, exec_mask=None, loc=None, ip=None):
    """Create amdgcn.cmpi."""
    op = _create_inst_op(
        "amdgcn.cmpi", opcode, outs=[dest, exec_mask], ins=[lhs, rhs], loc=loc, ip=ip
    )
    return op.results[0]


def int_to_offset_value(
    offset: Union[int, _ods_ir.Value], loc=None, ip=None
) -> _ods_ir.Value:
    """Convert an integer offset to an arith.constant Value, or return the Value as-is."""
    if isinstance(offset, int):
        from aster.dialects import arith

        ctx = _ods_ir.Context.current
        int_type = _ods_ir.IntegerType.get_signless(32, ctx)
        return arith.constant(int_type, offset, loc=loc, ip=ip)
    return offset
