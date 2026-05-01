# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""EmitC emission context: emits ``emitc.*`` ops instead of arith/func/scf."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Union

from aster import ir
from aster.dialects import emitc

from .context import EmissionContext, register_context
from .semantic import SemanticAnalyzer
from .types import ASTType, ASTValue, IndexType, ScalarType, i1

if TYPE_CHECKING:
    from .context import EmitterContext
    from .table import ASTEmitterProtocol, EmitterTable


# ---------------------------------------------------------------------------
# Class self sentinel.
# ---------------------------------------------------------------------------


class _ClassSelf:
    """Sentinel bound to ``self`` inside emitc class method bodies.

    Holds the field-name-to-``ASTType`` mapping so that ``self.x`` can
    be lowered to ``emitc.get_field @x``.
    """

    def __init__(self, fields: dict[str, ASTType]):
        self.fields = fields


# ---------------------------------------------------------------------------
# EmitC comparison predicate enum (matches upstream CmpPredicate).
# ---------------------------------------------------------------------------

_EMITC_CMP_EQ = 0
_EMITC_CMP_NE = 1
_EMITC_CMP_LT = 2
_EMITC_CMP_LE = 3
_EMITC_CMP_GT = 4
_EMITC_CMP_GE = 5

_CMP_MAP: dict[type[ast.cmpop], int] = {
    ast.Eq: _EMITC_CMP_EQ,
    ast.NotEq: _EMITC_CMP_NE,
    ast.Lt: _EMITC_CMP_LT,
    ast.LtE: _EMITC_CMP_LE,
    ast.Gt: _EMITC_CMP_GT,
    ast.GtE: _EMITC_CMP_GE,
}


# ---------------------------------------------------------------------------
# Binary-op emitters.
# ---------------------------------------------------------------------------


def _emit_add(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Add, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.AddOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_sub(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Sub, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.SubOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_mult(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Mult, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.MulOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_div(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Div, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.DivOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_mod(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.Mod, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.RemOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_bitand(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitAnd, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseAndOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_bitor(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitOr, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseOrOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_bitxor(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.BitXor, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseXorOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_lshift(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.LShift, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseLeftShiftOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


def _emit_rshift(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
    lhs, rhs, rty = _coerce_pair(ectx, ast.RShift, lhs, rhs)
    mlir_ty = rty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseRightShiftOp(
        mlir_ty, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, rty)


# ---------------------------------------------------------------------------
# Unary-op emitters.
# ---------------------------------------------------------------------------


def _emit_uadd(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    return operand


def _emit_usub(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    mlir_ty = ty.to_mlir_type(ectx.ctx)
    ir_val = emitc.UnaryMinusOp(
        mlir_ty, operand.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, ty)


def _emit_invert(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    mlir_ty = ty.to_mlir_type(ectx.ctx)
    ir_val = emitc.BitwiseNotOp(
        mlir_ty, operand.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, ty)


def _emit_not(ectx: EmitterContext, operand: ASTValue) -> ASTValue:
    ty = operand.ast_type
    mlir_ty = ty.to_mlir_type(ectx.ctx)
    ir_val = emitc.LogicalNotOp(
        mlir_ty, operand.ir_value, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, ty)


# ---------------------------------------------------------------------------
# Comparison emitters.
# ---------------------------------------------------------------------------


def _make_cmp_emitter(op: type[ast.cmpop]):
    """Return a comparison emitter for *op*."""
    pred_val = _CMP_MAP[op]

    def _emit_cmp(ectx: EmitterContext, lhs: ASTValue, rhs: ASTValue) -> ASTValue:
        lhs, rhs, _ = _coerce_pair(ectx, ast.Add, lhs, rhs)
        i1_mlir = i1.to_mlir_type(ectx.ctx)
        i64_mlir = ir.IntegerType.get_signless(64, ectx.ctx)
        pred_attr = ir.IntegerAttr.get(i64_mlir, pred_val)
        ir_val = emitc.CmpOp(
            i1_mlir, pred_attr, lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip
        ).result
        return ASTValue(ir_val, i1)

    return _emit_cmp


# ---------------------------------------------------------------------------
# Function / return emitters.
# ---------------------------------------------------------------------------


def _emit_function(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    arg_types: list[ASTType],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``emitc.func`` operation."""
    mlir_arg_types = [t.to_mlir_type(ectx.ctx) for t in arg_types]
    mlir_ret_types = [t.to_mlir_type(ectx.ctx) for t in ectx.return_types]

    func_type = ir.FunctionType.get(mlir_arg_types, mlir_ret_types)
    fn_op = emitc.FuncOp(
        node.name, ir.TypeAttr.get(func_type), loc=ectx.loc, ip=ectx.ip
    )

    arg_locs = [ectx.loc] * len(mlir_arg_types)
    entry_block = ir.Block.create_at_start(fn_op.body, mlir_arg_types, arg_locs)

    for i, arg_node in enumerate(node.args.args):
        block_arg = entry_block.arguments[i]
        ectx.scope_stack.bind_local(arg_node.arg, ASTValue(block_arg, arg_types[i]))

    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(entry_block)
    try:
        emitter.visit_stmts(node.body)
    finally:
        ectx.ip = saved_ip


def _emit_return(ectx: EmitterContext, values: list[ASTValue]) -> None:
    """Emit an ``emitc.return`` operation."""
    if values:
        emitc.ReturnOp(operand=values[0].ir_value, loc=ectx.loc, ip=ectx.ip)
    else:
        emitc.ReturnOp(loc=ectx.loc, ip=ectx.ip)


# ---------------------------------------------------------------------------
# If / for emitters.
# ---------------------------------------------------------------------------


def _emit_if(
    ectx: EmitterContext,
    cond: ASTValue,
    then_body: list[ast.stmt],
    else_body: list[ast.stmt],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``emitc.if`` with then/else regions."""
    cond_val = _coerce_to_i1(ectx, cond)

    if_op = emitc.IfOp(cond_val.ir_value, loc=ectx.loc, ip=ectx.ip)

    # Then region.
    then_block = ir.Block.create_at_start(if_op.thenRegion, [])
    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(then_block)
    try:
        emitter.visit_stmts(then_body)
        emitc.YieldOp(loc=ectx.loc, ip=ectx.ip)
    finally:
        ectx.ip = saved_ip

    # Else region (always present in emitc.if).
    else_block = ir.Block.create_at_start(if_op.elseRegion, [])
    if else_body:
        ectx.ip = ir.InsertionPoint(else_block)
        try:
            emitter.visit_stmts(else_body)
            emitc.YieldOp(loc=ectx.loc, ip=ectx.ip)
        finally:
            ectx.ip = saved_ip
    else:
        emitc.YieldOp(loc=ectx.loc, ip=ir.InsertionPoint(else_block))


def _emit_for(
    ectx: EmitterContext,
    target_name: str,
    lb: ASTValue,
    ub: ASTValue,
    step: ASTValue,
    body: list[ast.stmt],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``emitc.for`` loop.

    Unlike ``scf.for``, ``emitc.for`` does not require index-typed
    operands -- it works with any integer type.
    """
    for_op = emitc.ForOp(
        lb.ir_value, ub.ir_value, step.ir_value, loc=ectx.loc, ip=ectx.ip
    )

    iv_type = lb.ast_type
    iv_mlir = iv_type.to_mlir_type(ectx.ctx)
    body_block = ir.Block.create_at_start(for_op.region, [iv_mlir], [ectx.loc])
    iv = body_block.arguments[0]

    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(body_block)
    ectx.scope_stack.push_scope()
    try:
        ectx.scope_stack.bind_local(target_name, ASTValue(iv, iv_type))
        emitter.visit_stmts(body)
        emitc.YieldOp(loc=ectx.loc, ip=ectx.ip)
    finally:
        ectx.scope_stack.pop_scope()
        ectx.ip = saved_ip


# ---------------------------------------------------------------------------
# Class emission.
# ---------------------------------------------------------------------------


def _emit_class(
    ectx: EmitterContext,
    node: ast.ClassDef,
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``emitc.class`` from a Python ``class`` definition."""
    class_op = emitc.ClassOp(node.name, loc=ectx.loc, ip=ectx.ip)
    body_block = ir.Block.create_at_start(class_op.body, [])
    body_ip = ir.InsertionPoint(body_block)

    # First pass: collect fields from annotated assignments.
    fields: dict[str, ASTType] = {}
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        field_name = stmt.target.id
        field_type = emitter.resolve_type_annotation(stmt.annotation)
        fields[field_name] = field_type
        mlir_ty = field_type.to_mlir_type(ectx.ctx)
        emitc.FieldOp(field_name, mlir_ty, loc=ectx.loc, ip=body_ip)

    # Second pass: emit methods.
    self_sentinel = _ClassSelf(fields)
    for stmt in node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        _emit_method(ectx, stmt, self_sentinel, body_ip, emitter)


def _emit_method(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    self_sentinel: _ClassSelf,
    class_ip: ir.InsertionPoint,
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``emitc.func`` for a class method, stripping ``self``."""
    # Strip ``self`` from the argument list.
    all_args = node.args.args
    if not all_args or all_args[0].arg != "self":
        raise TypeError(
            f"class method '{node.name}' must have 'self' as its first parameter"
        )
    method_args = all_args[1:]

    # Resolve argument types.
    arg_types: list[ASTType] = []
    for arg in method_args:
        if arg.annotation is None:
            raise TypeError(
                f"argument '{arg.arg}' of method '{node.name}' has no type annotation"
            )
        arg_types.append(emitter.resolve_type_annotation(arg.annotation))

    # Resolve return types.
    ret_types: list[ASTType] = []
    if node.returns is not None:
        ret_types = [emitter.resolve_type_annotation(node.returns)]

    mlir_arg_types = [t.to_mlir_type(ectx.ctx) for t in arg_types]
    mlir_ret_types = [t.to_mlir_type(ectx.ctx) for t in ret_types]

    func_type = ir.FunctionType.get(mlir_arg_types, mlir_ret_types)
    fn_op = emitc.FuncOp(
        node.name, ir.TypeAttr.get(func_type), loc=ectx.loc, ip=class_ip
    )

    arg_locs = [ectx.loc] * len(mlir_arg_types)
    entry_block = ir.Block.create_at_start(fn_op.body, mlir_arg_types, arg_locs)

    saved_ip = ectx.ip
    saved_return = ectx.return_types
    ectx.ip = ir.InsertionPoint(entry_block)
    ectx.return_types = ret_types
    ectx.scope_stack.push_scope()
    try:
        # Bind ``self`` to the sentinel.
        ectx.scope_stack.bind_local("self", self_sentinel)
        # Bind remaining args.
        for i, arg_node in enumerate(method_args):
            block_arg = entry_block.arguments[i]
            ectx.scope_stack.bind_local(arg_node.arg, ASTValue(block_arg, arg_types[i]))
        emitter.visit_stmts(node.body)
    finally:
        ectx.scope_stack.pop_scope()
        ectx.ip = saved_ip
        ectx.return_types = saved_return


def _emit_self_attr(ectx: EmitterContext, obj: Any, attr_name: str) -> ASTValue:
    """Emit ``emitc.get_field`` for ``self.x`` access."""
    assert isinstance(obj, _ClassSelf)
    if attr_name not in obj.fields:
        raise AttributeError(f"class has no field '{attr_name}'")
    field_type = obj.fields[attr_name]
    mlir_ty = field_type.to_mlir_type(ectx.ctx)
    ir_val = emitc.GetFieldOp(mlir_ty, attr_name, loc=ectx.loc, ip=ectx.ip).result
    return ASTValue(ir_val, field_type)


# ---------------------------------------------------------------------------
# EmitC semantic analyser.
# ---------------------------------------------------------------------------


class EmitCSemantic:
    """Semantic analyser that materialises constants via ``emitc.constant``."""

    def infer_binop_type(
        self, op: type[ast.operator], lhs: ASTType, rhs: ASTType
    ) -> ASTType:
        if isinstance(lhs, IndexType) and isinstance(rhs, IndexType):
            return IndexType()
        if not isinstance(lhs, ScalarType) or not isinstance(rhs, ScalarType):
            raise TypeError(f"binary op on unsupported types: {lhs}, {rhs}")
        if lhs.is_float or rhs.is_float:
            width = max(
                lhs.width if lhs.is_float else 0,
                rhs.width if rhs.is_float else 0,
            )
            return ScalarType(width=width, is_float=True)
        return ScalarType(
            width=max(lhs.width, rhs.width),
            is_float=False,
            is_signed=lhs.is_signed or rhs.is_signed,
        )

    def coerce(
        self, ectx: EmitterContext, value: ASTValue, target: ASTType
    ) -> ASTValue:
        if value.ast_type == target:
            return value
        mlir_ty = target.to_mlir_type(ectx.ctx)
        ir_val = emitc.CastOp(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip).result
        return ASTValue(ir_val, target)

    def check_return(self, declared: list[ASTType], actual: list[ASTType]) -> None:
        if len(declared) != len(actual):
            raise TypeError(
                f"expected {len(declared)} return values, got {len(actual)}"
            )
        for i, (d, a) in enumerate(zip(declared, actual)):
            if d != a:
                raise TypeError(f"return value {i}: expected {d}, got {a}")

    def materialize_constant(
        self,
        ectx: EmitterContext,
        value: Union[int, float],
        target: ASTType,
    ) -> ASTValue:
        mlir_ty = target.to_mlir_type(ectx.ctx)
        if isinstance(value, float) or (
            isinstance(target, ScalarType) and target.is_float
        ):
            attr = ir.FloatAttr.get(mlir_ty, float(value))
        else:
            attr = ir.IntegerAttr.get(mlir_ty, int(value))
        ir_val = emitc.ConstantOp(mlir_ty, attr, loc=ectx.loc, ip=ectx.ip).result
        return ASTValue(ir_val, target)


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


def _coerce_to_i1(ectx: EmitterContext, value: ASTValue) -> ASTValue:
    """Ensure *value* is i1 (boolean)."""
    if isinstance(value.ast_type, ScalarType) and value.ast_type == i1:
        return value
    return ectx.semantic.coerce(ectx, value, i1)


# ---------------------------------------------------------------------------
# Table population and context registration.
# ---------------------------------------------------------------------------


def _populate_emitc_table(table: EmitterTable) -> None:
    """Register all EmitC emitter handlers on *table*."""
    table.register_binop(ast.Add, _emit_add)
    table.register_binop(ast.Sub, _emit_sub)
    table.register_binop(ast.Mult, _emit_mult)
    table.register_binop(ast.Div, _emit_div)
    table.register_binop(ast.Mod, _emit_mod)
    table.register_binop(ast.BitAnd, _emit_bitand)
    table.register_binop(ast.BitOr, _emit_bitor)
    table.register_binop(ast.BitXor, _emit_bitxor)
    table.register_binop(ast.LShift, _emit_lshift)
    table.register_binop(ast.RShift, _emit_rshift)

    table.register_unaryop(ast.UAdd, _emit_uadd)
    table.register_unaryop(ast.USub, _emit_usub)
    table.register_unaryop(ast.Invert, _emit_invert)
    table.register_unaryop(ast.Not, _emit_not)

    for cmpop in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE):
        table.register_cmpop(cmpop, _make_cmp_emitter(cmpop))

    table.register_function_emitter(_emit_function)
    table.register_return_emitter(_emit_return)
    table.register_if_emitter(_emit_if)
    table.register_for_emitter(_emit_for)
    table.register_class_emitter(_emit_class)
    table.register_attribute_emitter(_ClassSelf, _emit_self_attr)


class EmitCContext(EmissionContext):
    """Emission context that produces ``emitc.*`` ops."""

    def configure_table(self, parent: EmitterTable) -> EmitterTable:
        from .table import EmitterTable as ET

        table = ET(parent=parent)
        _populate_emitc_table(table)
        return table

    def configure_semantic(self, parent: SemanticAnalyzer) -> SemanticAnalyzer:
        return EmitCSemantic()


register_context("emitc", EmitCContext)
