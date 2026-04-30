# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emitters for ``arith`` dialect: binary ops, unary ops, comparisons."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from aster.dialects import arith
from aster.dialects._arith_enum_gen import CmpFPredicate, CmpIPredicate

from .types import ASTValue, IndexType, ScalarType, i1

if TYPE_CHECKING:
    from .context import EmitterContext
    from .table import EmitterTable


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _coerce_pair(
    ectx: EmitterContext,
    op: type[ast.operator],
    lhs: ASTValue,
    rhs: ASTValue,
) -> tuple[ASTValue, ASTValue, ScalarType | IndexType]:
    """Promote *lhs* and *rhs* to a common type via the semantic layer."""
    result_type = ectx.semantic.infer_binop_type(op, lhs.ast_type, rhs.ast_type)
    lhs = ectx.semantic.coerce(ectx, lhs, result_type)
    rhs = ectx.semantic.coerce(ectx, rhs, result_type)
    return lhs, rhs, result_type


def _is_float(ty: ScalarType | IndexType) -> bool:
    return isinstance(ty, ScalarType) and ty.is_float


def _is_unsigned(ty: ScalarType | IndexType) -> bool:
    return isinstance(ty, ScalarType) and not ty.is_float and not ty.is_signed


# ---------------------------------------------------------------------------
# Binary-op emitters.
# ---------------------------------------------------------------------------


def _emit_add(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Add, lhs, rhs)
    if _is_float(rty):
        ir_val = arith.addf(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        ir_val = arith.addi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_sub(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Sub, lhs, rhs)
    if _is_float(rty):
        ir_val = arith.subf(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        ir_val = arith.subi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_mult(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Mult, lhs, rhs)
    if _is_float(rty):
        ir_val = arith.mulf(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        ir_val = arith.muli(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_div(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Div, lhs, rhs)
    if _is_float(rty):
        ir_val = arith.divf(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    elif _is_unsigned(rty):
        ir_val = arith.divui(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        ir_val = arith.divsi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_floordiv(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.FloorDiv, lhs, rhs)
    if _is_float(rty):
        raise TypeError(
            "floor division (//) on floats is not supported by the arith dialect; "
            "use true division (/) combined with math.floor via an @escape function"
        )
    if _is_unsigned(rty):
        # Unsigned division already truncates toward zero, which is floor for positives.
        ir_val = arith.divui(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        # arith.floordivsi matches Python // semantics (rounds toward -inf).
        ir_val = arith.floordivsi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_mod(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Mod, lhs, rhs)
    if _is_float(rty):
        ir_val = arith.remf(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    elif _is_unsigned(rty):
        ir_val = arith.remui(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        ir_val = arith.remsi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_lshift(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.LShift, lhs, rhs)
    ir_val = arith.shli(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_rshift(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.RShift, lhs, rhs)
    ir_val = arith.shrsi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_bitand(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitAnd, lhs, rhs)
    ir_val = arith.andi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_bitor(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitOr, lhs, rhs)
    ir_val = arith.ori(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


def _emit_bitxor(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitXor, lhs, rhs)
    ir_val = arith.xori(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, rty)


# ---------------------------------------------------------------------------
# Unary-op emitters.
# ---------------------------------------------------------------------------


def _emit_uadd(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    return operand


def _emit_usub(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    if isinstance(ty, ScalarType) and ty.is_float:
        ir_val = arith.negf(operand.ir_value, loc=ectx.loc, ip=ectx.ip)
        return ASTValue(ir_val, ty)
    # Integer negation: 0 - x.
    zero = ectx.semantic.materialize_constant(ectx, 0, ty)
    ir_val = arith.subi(zero.ir_value, operand.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, ty)


def _emit_invert(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    # Bitwise NOT: x ^ -1.
    minus_one = ectx.semantic.materialize_constant(ectx, -1, ty)
    ir_val = arith.xori(operand.ir_value, minus_one.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, ty)


def _emit_not(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    if isinstance(ty, ScalarType) and ty == i1:
        # Flip the single bit: x ^ 1.
        one = ectx.semantic.materialize_constant(ectx, 1, i1)
        ir_val = arith.xori(operand.ir_value, one.ir_value, loc=ectx.loc, ip=ectx.ip)
        return ASTValue(ir_val, i1)
    if isinstance(ty, (ScalarType, IndexType)):
        # Produce i1: true when the operand is zero.
        zero = ectx.semantic.materialize_constant(ectx, 0, ty)
        ir_val = arith.cmpi(
            CmpIPredicate.eq,
            operand.ir_value,
            zero.ir_value,
            loc=ectx.loc,
            ip=ectx.ip,
        )
        return ASTValue(ir_val, i1)
    raise TypeError(f"'not' operator not supported for type {ty}")


# ---------------------------------------------------------------------------
# Comparison emitters.
# ---------------------------------------------------------------------------

_CMPI_MAP: dict[type[ast.cmpop], CmpIPredicate] = {
    ast.Eq: CmpIPredicate.eq,
    ast.NotEq: CmpIPredicate.ne,
    ast.Lt: CmpIPredicate.slt,
    ast.LtE: CmpIPredicate.sle,
    ast.Gt: CmpIPredicate.sgt,
    ast.GtE: CmpIPredicate.sge,
}

_CMPF_MAP: dict[type[ast.cmpop], CmpFPredicate] = {
    ast.Eq: CmpFPredicate.OEQ,
    ast.NotEq: CmpFPredicate.ONE,
    ast.Lt: CmpFPredicate.OLT,
    ast.LtE: CmpFPredicate.OLE,
    ast.Gt: CmpFPredicate.OGT,
    ast.GtE: CmpFPredicate.OGE,
}


def _make_cmp_emitter(
    op: type[ast.cmpop],
):
    """Return a comparison emitter for *op*."""

    def _emit_cmp(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
        # Pass ast.Add as a proxy op: infer_binop_type only needs to determine
        # the common input type; the comparison result is always i1.
        lhs, rhs, rty = _coerce_pair(ectx, ast.Add, lhs, rhs)
        if _is_float(rty):
            pred = _CMPF_MAP[op]
            ir_val = arith.cmpf(
                pred, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
            )
        else:
            pred = _CMPI_MAP[op]
            ir_val = arith.cmpi(
                pred, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
            )
        return ASTValue(ir_val, i1)

    return _emit_cmp


# ---------------------------------------------------------------------------
# Registration helper.
# ---------------------------------------------------------------------------


def register_arith_emitters(table: EmitterTable) -> None:
    """Populate *table* with the default arith emitters."""
    table.register_binop(ast.Add, _emit_add)
    table.register_binop(ast.Sub, _emit_sub)
    table.register_binop(ast.Mult, _emit_mult)
    table.register_binop(ast.Div, _emit_div)
    table.register_binop(ast.FloorDiv, _emit_floordiv)
    table.register_binop(ast.Mod, _emit_mod)
    table.register_binop(ast.LShift, _emit_lshift)
    table.register_binop(ast.RShift, _emit_rshift)
    table.register_binop(ast.BitAnd, _emit_bitand)
    table.register_binop(ast.BitOr, _emit_bitor)
    table.register_binop(ast.BitXor, _emit_bitxor)

    table.register_unaryop(ast.UAdd, _emit_uadd)
    table.register_unaryop(ast.USub, _emit_usub)
    table.register_unaryop(ast.Invert, _emit_invert)
    table.register_unaryop(ast.Not, _emit_not)

    for cmpop in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE):
        table.register_cmpop(cmpop, _make_cmp_emitter(cmpop))
