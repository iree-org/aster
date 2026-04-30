# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emitters for ``scf.if`` and ``scf.for`` via ``ir.Operation.create``."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from aster import ir
from aster.dialects import arith

from .types import ASTValue, IndexType, ScalarType, i1

if TYPE_CHECKING:
    from .context import EmitterContext
    from .table import ASTEmitterProtocol, EmitterTable


# ---------------------------------------------------------------------------
# scf.if
# ---------------------------------------------------------------------------


def emit_if(
    ectx: EmitterContext,
    cond: ASTValue,
    then_body: list[ast.stmt],
    else_body: list[ast.stmt],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``scf.if`` with then/else regions.

    For simplicity the initial implementation does not produce results
    from the ``scf.if`` (the bodies are executed for side effects only).
    Loop-carried / if-carried values can be added later.
    """
    cond_val = _coerce_to_i1(ectx, cond)

    has_else = bool(else_body)
    num_regions = 2 if has_else else 1

    if_op = ir.Operation.create(
        "scf.if",
        results=[],
        operands=[cond_val.ir_value],
        regions=num_regions,
        loc=ectx.loc,
        ip=ectx.ip,
    )

    # Then region.
    then_block = ir.Block.create_at_start(if_op.regions[0], [])
    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(then_block)
    try:
        emitter.visit_stmts(then_body)
        ir.Operation.create("scf.yield", operands=[], loc=ectx.loc, ip=ectx.ip)
    finally:
        ectx.ip = saved_ip

    # Else region.
    if has_else:
        else_block = ir.Block.create_at_start(if_op.regions[1], [])
        ectx.ip = ir.InsertionPoint(else_block)
        try:
            emitter.visit_stmts(else_body)
            ir.Operation.create("scf.yield", operands=[], loc=ectx.loc, ip=ectx.ip)
        finally:
            ectx.ip = saved_ip


# ---------------------------------------------------------------------------
# scf.for
# ---------------------------------------------------------------------------


def emit_for(
    ectx: EmitterContext,
    target_name: str,
    lb: ASTValue,
    ub: ASTValue,
    step: ASTValue,
    body: list[ast.stmt],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit an ``scf.for`` loop.

    The induction variable is bound to *target_name* inside the loop
    body. Currently no iter_args (loop-carried values) are supported.
    """
    idx_ty = IndexType()

    lb = _coerce_to_index(ectx, lb)
    ub = _coerce_to_index(ectx, ub)
    step = _coerce_to_index(ectx, step)

    for_op = ir.Operation.create(
        "scf.for",
        results=[],
        operands=[lb.ir_value, ub.ir_value, step.ir_value],
        regions=1,
        loc=ectx.loc,
        ip=ectx.ip,
    )

    idx_mlir = idx_ty.to_mlir_type(ectx.ctx)
    body_block = ir.Block.create_at_start(for_op.regions[0], [idx_mlir], [ectx.loc])

    iv = body_block.arguments[0]

    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(body_block)
    ectx.scope_stack.push_scope()
    try:
        ectx.scope_stack.bind_local(target_name, ASTValue(iv, idx_ty))
        emitter.visit_stmts(body)
        ir.Operation.create("scf.yield", operands=[], loc=ectx.loc, ip=ectx.ip)
    finally:
        ectx.scope_stack.pop_scope()
        ectx.ip = saved_ip


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _coerce_to_i1(ectx: EmitterContext, value: ASTValue) -> ASTValue:
    """Ensure *value* is i1 (boolean).

    Insert truncation if needed.
    """
    if isinstance(value.ast_type, ScalarType) and value.ast_type == i1:
        return value
    if isinstance(value.ast_type, ScalarType) and not value.ast_type.is_float:
        # Integer -> i1 via trunci.
        i1_ty = i1.to_mlir_type(ectx.ctx)
        ir_val = arith.trunci(i1_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
        return ASTValue(ir_val, i1)
    raise TypeError(f"cannot coerce {value.ast_type} to i1 for scf.if condition")


def _coerce_to_index(ectx: EmitterContext, value: ASTValue) -> ASTValue:
    """Ensure *value* is index type.

    Insert index_cast if needed.
    """
    if isinstance(value.ast_type, IndexType):
        return value
    idx_ty = IndexType()
    mlir_ty = idx_ty.to_mlir_type(ectx.ctx)
    ir_val = arith.index_cast(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
    return ASTValue(ir_val, idx_ty)


def register_scf_emitters(table: EmitterTable) -> None:
    """Populate *table* with the default scf emitters."""
    table.register_if_emitter(emit_if)
    table.register_for_emitter(emit_for)
