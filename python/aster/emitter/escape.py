# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Escape hatches: ``@escape`` decorator and ``reflect()`` inline escape."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from aster import ir

from .context import EmitterContext
from .types import ASTValue


def escape(fn: Callable) -> Callable:
    """Mark *fn* as a compile-time function.

    When the emitter encounters a call to an escaped function it
    executes it in Python rather than lowering it to MLIR.
    """
    fn._aster_escape = True  # type: ignore[attr-defined]
    return fn


def is_escaped(fn: Any) -> bool:
    """Return whether *fn* is marked with ``@escape``."""
    return getattr(fn, "_aster_escape", False) is True


def reflect(fn: Callable, *args: Any) -> Any:
    """Marker for inline compile-time escape.

    At the call site the emitter resolves arguments (``ASTValue`` or
    plain Python) and invokes *fn*.  This function itself is never
    called at runtime -- it is only used as a sentinel that the emitter
    recognises.
    """
    raise RuntimeError("aster.reflect() must only be called inside @aster.jit() code")


# Sentinel identity so the emitter can detect ``aster.reflect`` in call nodes.
_REFLECT_SENTINEL = reflect


def execute_reflect(
    ectx: EmitterContext,
    fn: Callable,
    resolved_args: list[Any],
) -> Any:
    """Actually call *fn* with *resolved_args*, injecting the context if requested."""
    inject_ctx = False
    try:
        sig = inspect.signature(fn)
        for param in sig.parameters.values():
            annotation = param.annotation
            if annotation is EmitterContext or (
                isinstance(annotation, str) and annotation == "EmitterContext"
            ):
                inject_ctx = True
                break
    except (ValueError, TypeError):
        # Built-in functions and C extensions have no inspectable signature;
        # they never receive the context injection.
        pass

    if inject_ctx:
        result = fn(ectx, *resolved_args)
    else:
        result = fn(*resolved_args)

    return _wrap_result(result, ectx)


def execute_escape(
    ectx: EmitterContext,
    fn: Callable,
    resolved_args: list[Any],
) -> Any:
    """Execute an ``@escape``-decorated function at compile time."""
    return execute_reflect(ectx, fn, resolved_args)


def _wrap_result(result: Any, ectx: EmitterContext) -> Any:
    """Wrap a raw ``ir.Value`` coming back from an escape into an ``ASTValue``."""
    if result is None:
        return None
    if isinstance(result, ASTValue):
        return result
    if isinstance(result, ir.Value):
        # Best-effort: wrap with the MLIR type but no frontend type info.
        from .types import ScalarType

        mlir_type = result.type
        # Try to recover a ScalarType from the MLIR type.
        try:
            if ir.IntegerType.isinstance(mlir_type):
                int_ty = ir.IntegerType(mlir_type)
                ast_type = ScalarType(width=int_ty.width)
            elif ir.IndexType.isinstance(mlir_type):
                from .types import IndexType

                ast_type = IndexType()
            elif ir.F16Type.isinstance(mlir_type):
                ast_type = ScalarType(width=16, is_float=True)
            elif ir.F32Type.isinstance(mlir_type):
                ast_type = ScalarType(width=32, is_float=True)
            elif ir.F64Type.isinstance(mlir_type):
                ast_type = ScalarType(width=64, is_float=True)
            else:
                ast_type = ScalarType(width=0)
        except Exception:
            ast_type = ScalarType(width=0)
        return ASTValue(result, ast_type)
    return result
