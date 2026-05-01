# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pattern emission context: emits ``pattern.*`` ops for rewrite patterns.

Uses EmitC ops for expressions inside pattern bodies (since the pattern
dialect translates to C++ via EmitC interfaces).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any, Optional

from aster import ir
from aster.dialects import emitc
from aster.dialects import pattern as pattern_dialect

from .context import EmissionContext, register_context
from .emitc_emitter import (
    EmitCSemantic,
    _MethodReceiver,
    _populate_emitc_table,
)
from .types import ASTType, ASTValue

if TYPE_CHECKING:
    from .context import EmitterContext
    from .semantic import SemanticAnalyzer
    from .table import ASTEmitterProtocol, EmitterTable


# ---------------------------------------------------------------------------
# Pattern field sentinel.
# ---------------------------------------------------------------------------


class _PatternField:
    """Sentinel bound in the scope for each declared field.

    When the emitter resolves a field name (e.g. ``counter``), it finds
    this sentinel and the ``_PatternFieldEmitter`` attribute emitter
    creates a ``pattern.get_field`` op.
    """

    def __init__(self, name: str, ast_type: ASTType):
        self.name = name
        self.ast_type = ast_type


# ---------------------------------------------------------------------------
# Name emitter for _PatternField.
# ---------------------------------------------------------------------------


def _emit_field_name(ectx: EmitterContext, obj: Any) -> ASTValue:
    """Emit ``pattern.get_field`` when a field name is resolved via visit_Name."""
    assert isinstance(obj, _PatternField)
    mlir_ty = obj.ast_type.to_mlir_type(ectx.ctx)
    ir_val = pattern_dialect.GetFieldOp(
        mlir_ty, obj.name, loc=ectx.loc, ip=ectx.ip
    ).result
    return ASTValue(ir_val, obj.ast_type)


# ---------------------------------------------------------------------------
# Pattern function emitter.
# ---------------------------------------------------------------------------


def _emit_pattern_function(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    arg_types: list[ASTType],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit a ``pattern.rewrite_pattern`` from a decorated function.

    The function must have pattern metadata stored on the emitter context
    via ``_pattern_meta``.
    """
    meta = getattr(ectx, "_pattern_meta", None)
    if meta is None:
        raise TypeError("pattern function emitter invoked without pattern metadata")

    benefit: int = meta["benefit"]
    op_name: str = meta["op"]
    fields: list[tuple[str, ASTType]] = meta["fields"]

    rp = pattern_dialect.RewritePatternOp(
        node.name, benefit, op_name, loc=ectx.loc, ip=ectx.ip
    )

    # Populate the fields region if there are any fields.
    if fields:
        fields_block = ir.Block.create_at_start(rp.fieldsRegion, [])
        fields_ip = ir.InsertionPoint(fields_block)
        field_results: list[ir.Value] = []
        for fname, ftype in fields:
            mlir_ty = ftype.to_mlir_type(ectx.ctx)
            field_op = pattern_dialect.FieldOp(
                mlir_ty, fname, loc=ectx.loc, ip=fields_ip
            )
            field_results.append(field_op.result)
        pattern_dialect.YieldOp(field_results, loc=ectx.loc, ip=fields_ip)

    # Populate the body region.
    body_block = ir.Block.create_at_start(rp.bodyRegion, [])

    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(body_block)

    # Bind field names in scope as _PatternField sentinels so that bare
    # references like ``counter`` resolve to pattern.get_field.
    ectx.scope_stack.push_scope()
    try:
        for fname, ftype in fields:
            ectx.scope_stack.bind_local(fname, _PatternField(fname, ftype))

        # Bind ``op`` and ``rewriter`` as _MethodReceiver sentinels so that
        # method calls like ``rewriter.eraseOp(op)`` emit
        # ``pattern.method_call``.
        param_types = {"op": op_name, "rewriter": "PatternRewriter"}
        for arg in node.args.args:
            param_name = arg.arg
            cpp_type = param_types.get(param_name, param_name)
            opaque_ty = ir.Type.parse(f'!emitc.opaque<"{cpp_type}">', context=ectx.ctx)
            lit_val = emitc.LiteralOp(
                opaque_ty, param_name, loc=ectx.loc, ip=ectx.ip
            ).result
            ectx.scope_stack.bind_local(
                param_name, _MethodReceiver(param_name, lit_val)
            )

        emitter.visit_stmts(node.body)
    finally:
        ectx.scope_stack.pop_scope()
        ectx.ip = saved_ip


# ---------------------------------------------------------------------------
# Rewrite (action) handler.
# ---------------------------------------------------------------------------


def _emit_rewrite_action(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    arg_types: list[ASTType],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit ``pattern.action`` from ``def rewrite(cond): ...``.

    The single parameter name is looked up in the current scope to
    obtain the condition value.
    """
    if len(node.args.args) != 1:
        raise TypeError("def rewrite() must have exactly one parameter (the condition)")

    cond_name = node.args.args[0].arg
    try:
        cond = ectx.scope_stack.resolve(cond_name)
    except KeyError:
        raise NameError(
            f"condition name '{cond_name}' is not defined in the current scope"
        ) from None

    if not isinstance(cond, ASTValue):
        raise TypeError(
            f"condition '{cond_name}' must be an IR value, got {type(cond).__name__}"
        )

    action_op = pattern_dialect.ActionOp(cond.ir_value, loc=ectx.loc, ip=ectx.ip)
    action_block = ir.Block.create_at_start(action_op.bodyRegion, [])

    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(action_block)
    ectx.scope_stack.push_scope()
    try:
        emitter.visit_stmts(node.body)
        pattern_dialect.YieldOp([], loc=ectx.loc, ip=ectx.ip)
    finally:
        ectx.scope_stack.pop_scope()
        ectx.ip = saved_ip


# ---------------------------------------------------------------------------
# Combined function emitter dispatcher.
# ---------------------------------------------------------------------------


def _pattern_function_dispatcher(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    arg_types: list[ASTType],
    emitter: ASTEmitterProtocol,
) -> None:
    """Dispatch: ``def rewrite(cond):`` -> action, else -> pattern function."""
    if node.name == "rewrite":
        _emit_rewrite_action(ectx, node, arg_types, emitter)
        return
    _emit_pattern_function(ectx, node, arg_types, emitter)


# ---------------------------------------------------------------------------
# Table population and context.
# ---------------------------------------------------------------------------


def _populate_pattern_table(table: EmitterTable) -> None:
    """Register pattern-specific emitters on top of emitc handlers."""
    _populate_emitc_table(table)
    table.register_function_emitter(_pattern_function_dispatcher)
    table.register_name_emitter(_PatternField, _emit_field_name)


class PatternContext(EmissionContext):
    """Emission context that produces ``pattern.*`` ops.

    Inherits all EmitC binary/unary/comparison ops so that expressions
    inside pattern bodies compile to ``emitc.*`` ops for C++ translation.
    """

    def configure_table(self, parent: EmitterTable) -> EmitterTable:
        from .table import EmitterTable as ET

        table = ET(parent=parent)
        _populate_pattern_table(table)
        return table

    def configure_semantic(self, parent: SemanticAnalyzer) -> SemanticAnalyzer:
        return EmitCSemantic()


register_context("pattern", PatternContext)


# ---------------------------------------------------------------------------
# @pattern decorator (top-level, like @jit).
# ---------------------------------------------------------------------------


def _parse_field_decl(decl: str) -> tuple[str, ASTType]:
    """Parse a field declaration string like ``"counter: i32"``."""
    parts = decl.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid field declaration '{decl}', expected 'name: type'")
    name = parts[0].strip()
    type_name = parts[1].strip()

    from . import types as _types_mod

    val = getattr(_types_mod, type_name, None)
    if isinstance(val, ASTType):
        return name, val
    raise TypeError(f"unknown field type: {type_name}")


def pattern(
    benefit: int = 0,
    op: str = "",
    fields: Optional[list[str]] = None,
):
    """Decorator that captures a function as a rewrite pattern.

    Usage::

        @aster.pattern(benefit=1, op="MyOp", fields=["counter: i32"])
        def my_pat(op, rewriter):
            v = counter
            cond = v == 0
            def rewrite(cond):
                pass

        module = my_pat.emit()
    """

    def wrapper(fn):
        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except OSError as exc:
            raise TypeError(
                f"cannot retrieve source for '{fn.__name__}': {exc}"
            ) from exc
        tree = ast.parse(source)

        # Strip the @pattern(...) decorator from the AST so that
        # visit_FunctionDef does not treat it as a context block.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
                node.decorator_list = [
                    d for d in node.decorator_list if not _is_pattern_decorator(d)
                ]
                break

        parsed_fields: list[tuple[str, ASTType]] = []
        if fields:
            parsed_fields = [_parse_field_decl(f) for f in fields]

        from .table import EmitterTable

        table = EmitterTable()
        _populate_pattern_table(table)

        from .core import JITFunction

        jit_fn = JITFunction(
            name=fn.__name__,
            fn=fn,
            tree=tree,
            semantic=EmitCSemantic(),
            table=table,
        )
        jit_fn._pattern_meta = {
            "benefit": benefit,
            "op": op,
            "fields": parsed_fields,
        }
        return jit_fn

    return wrapper


def _is_pattern_decorator(deco: ast.AST) -> bool:
    """Return True if *deco* is ``@pattern(...)`` or ``@aster.pattern(...)``."""
    node = deco
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name) and node.id == "pattern":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "pattern":
        return True
    return False
