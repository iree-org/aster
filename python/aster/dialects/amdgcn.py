#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._amdgcn_ops_gen import *
from ._amdgcn_enum_gen import *
from ._ods_common import _cext as _ods_cext

_ods_ir = _ods_cext.ir

# Import register types from C++ bindings
from .._mlir_libs._amdgcn import (
    AGPRType,
    AGPRRangeType,
    SGPRType,
    SGPRRangeType,
    VGPRType,
    VGPRRangeType,
)

# Import API functions
from . import api

__all__ = [
    "AGPRType",
    "AGPRRangeType",
    "SGPRType",
    "SGPRRangeType",
    "VGPRType",
    "VGPRRangeType",
    "api",
]

from typing import Optional, Union


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


def vop2(opcode, dest, src0, src1, *, dst1=None, src2=None, loc=None, ip=None):
    """Create amdgcn.vop2 with correct segment sizes.

    The ODS-generated VOP2Op has _ODS_RESULT_SEGMENTS = [0, 0] (both optional), so the
    generic OpView.__init__ cannot infer resultSegmentSizes from just the result types.
    We use VOP2Op but pass the result types list that encodes the segment sizes
    implicitly (1 type for vdst0, 0 or 1 for dst1).
    """
    result_types = [dest.type]
    if dst1 is not None:
        result_types.append(dst1.type)
    return VOP2Op(
        opcode=opcode,
        vdst0=dest,
        src0=src0,
        src1=src1,
        dst1=dst1,
        src2=src2,
        results=result_types,
        loc=loc,
        ip=ip,
    ).results[0]


def int_to_offset_value(
    offset: Union[int, _ods_ir.Value], loc=None, ip=None
) -> _ods_ir.Value:
    """Convert an integer offset to an arith.constant Value, or return the Value as-is."""
    if isinstance(offset, int):
        from ..dialects import arith

        ctx = _ods_ir.Context.current
        int_type = _ods_ir.IntegerType.get_signless(32, ctx)
        return arith.constant(int_type, offset, loc=loc, ip=ip)
    return offset
