# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python bindings for the Layout dialect.

Usage:
    from aster.dialects import layout

    # Build a layout attribute from Python lists
    attr = layout.strided_layout([4, 8], [1, 4])           # flat
    attr = layout.strided_layout([(2, 2), (2, 4)],
                                 [(1, 4), (2, 8)])          # nested

    # Emit layout.linearize op
    offset = layout.linearize(coord, attr)
"""

from ._layout_ops_gen import *  # noqa: F401, F403
from ._layout_ops_gen import _Dialect, LinearizeOp, SwizzleOp  # noqa: F401
from ._ods_common import _cext as _ods_cext

_ods_ir = _ods_cext.ir


def _format_int_tuple_element(val) -> str:
    """Format an int or nested tuple for MLIR assembly."""
    if isinstance(val, int):
        return str(val)
    return "(" + ", ".join(_format_int_tuple_element(v) for v in val) + ")"


def _format_int_tuple_array(vals) -> str:
    """Format a top-level array for MLIR assembly."""
    return "[" + ", ".join(_format_int_tuple_element(v) for v in vals) + "]"


def strided_layout(shape, stride, *, ctx=None):
    """Build a #layout.strided_layout attribute from Python lists.

    Args:
        shape: list of ints or nested tuples for shape
        stride: list of ints or nested tuples for stride (same structure)
        ctx: MLIR context (uses default if not provided)

    Returns:
        A #layout.strided_layout<[shape] : [stride]> attribute.

    Examples:
        strided_layout([4, 8], [1, 4])              # col-major 4x8
        strided_layout([4, 8], [8, 1])              # row-major 4x8
        strided_layout([(2, 2), (2, 4)],
                       [(1, 4), (2, 8)])             # nested
    """
    if ctx is None:
        ctx = _ods_ir.Context.current
    asm = (
        f"#layout.strided_layout<{_format_int_tuple_array(shape)}"
        f" : {_format_int_tuple_array(stride)}>"
    )
    return _ods_ir.Attribute.parse(asm, ctx)


def linearize(coord, layout_attr, *, loc=None, ip=None):
    """Emit a layout.linearize op.

    Args:
        coord: index-typed SSA value (the logical coordinate)
        layout_attr: a #layout.strided_layout attribute
        loc: optional location
        ip: optional insertion point

    Returns:
        index-typed SSA value (the physical offset)
    """
    idx_type = _ods_ir.IndexType.get()
    return LinearizeOp(
        result=idx_type,
        coord=coord,
        layout=layout_attr,
        loc=loc,
        ip=ip,
    ).result


def swizzle(offset, *, bits, base, shift, loc=None, ip=None):
    """Emit a layout.swizzle op (XOR-based address permutation).

    result = offset ^ ((offset >> shift) & (((1 << bits) - 1) << base))

    Args:
        offset: index-typed SSA value
        bits: number of bits in the XOR mask
        base: bit position of the mask
        shift: right-shift amount before masking
    """
    idx_type = _ods_ir.IndexType.get()
    return SwizzleOp(
        result=idx_type,
        offset=offset,
        bits=bits,
        base=base,
        shift=shift,
        loc=loc,
        ip=ip,
    ).result
