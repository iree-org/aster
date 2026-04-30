# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emitters for ``func.func`` and ``func.return``."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from aster import ir
from aster.dialects import func as funcd

from .types import ASTType, ASTValue

if TYPE_CHECKING:
    from .context import EmitterContext
    from .table import ASTEmitterProtocol, EmitterTable


def emit_function(
    ectx: EmitterContext,
    node: ast.FunctionDef,
    arg_types: list[ASTType],
    emitter: ASTEmitterProtocol,
) -> None:
    """Emit a ``func.func`` operation from a ``FunctionDef`` AST node."""
    mlir_arg_types = [t.to_mlir_type(ectx.ctx) for t in arg_types]
    mlir_ret_types = [t.to_mlir_type(ectx.ctx) for t in ectx.return_types]

    func_type = ir.FunctionType.get(mlir_arg_types, mlir_ret_types)
    fn_op = funcd.FuncOp(
        node.name, func_type, visibility="public", loc=ectx.loc, ip=ectx.ip
    )

    arg_locs = [ectx.loc] * len(mlir_arg_types)
    entry_block = ir.Block.create_at_start(fn_op.body, mlir_arg_types, arg_locs)

    # Bind block arguments to parameter names in the current scope.
    for i, arg_node in enumerate(node.args.args):
        block_arg = entry_block.arguments[i]
        ectx.scope_stack.bind_local(arg_node.arg, ASTValue(block_arg, arg_types[i]))

    # Visit the function body with the entry block's insertion point.
    saved_ip = ectx.ip
    ectx.ip = ir.InsertionPoint(entry_block)
    try:
        emitter.visit_stmts(node.body)
    finally:
        ectx.ip = saved_ip


def emit_return(ectx: EmitterContext, values: list[ASTValue]) -> None:
    """Emit a ``func.return`` operation."""
    ir_values = [v.ir_value for v in values]
    funcd.ReturnOp(ir_values, loc=ectx.loc, ip=ectx.ip)


def register_func_emitters(table: EmitterTable) -> None:
    """Populate *table* with the default func emitters."""
    table.register_function_emitter(emit_function)
    table.register_return_emitter(emit_return)
