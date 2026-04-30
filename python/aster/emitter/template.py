# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Template functions: ``@template`` decorator, ``instantiate()``, and AST substitution."""

from __future__ import annotations

import ast
import copy
from typing import Any, Callable

from .types import ASTType, TemplateParam, TemplateParamKind


class TemplateFunction:
    """A generic function whose AST is stored unevaluated.

    Created by stacking ``@aster.template(...)`` on top of ``@aster.jit()``.
    """

    def __init__(self, jit_fn: Any, param_decls: dict[str, Any]):
        self.jit_fn = jit_fn
        self.params: dict[str, TemplateParam] = {
            name: TemplateParam.from_decl(name, decl)
            for name, decl in param_decls.items()
        }
        self._cache: dict[tuple[tuple[str, Any], ...], Any] = {}

    def __repr__(self) -> str:
        params_str = ", ".join(f"{p.name}={p.kind.value}" for p in self.params.values())
        return f"TemplateFunction({self.jit_fn.name}, {params_str})"


def template(**params: Any) -> Callable:
    """Decorator declaring template parameters.

    Usage::

        @aster.template(T=Type, N=int, op=Callable)
        @aster.jit()
        def kernel(a: T, b: T) -> T:
            ...

    The decorated object must be a ``JITFunction`` (already wrapped by
    ``@jit``).  Returns a ``TemplateFunction``.
    """

    def wrapper(jit_fn: Any) -> TemplateFunction:
        return TemplateFunction(jit_fn, params)

    return wrapper


def instantiate(template_fn: TemplateFunction, **bindings: Any) -> Any:
    """Substitute template parameters and return a concrete ``JITFunction``.

    Works both outside JIT (returns a new ``JITFunction``) and inside
    JIT (the emitter intercepts and emits a ``func.call``).
    """
    _validate_bindings(template_fn, bindings)

    cache_key = _make_cache_key(bindings)
    cached = template_fn._cache.get(cache_key)
    if cached is not None:
        return cached

    original_tree = copy.deepcopy(template_fn.jit_fn.tree)
    substituter = TemplateSubstituter(template_fn.params, bindings)
    new_tree = substituter.visit(original_tree)
    ast.fix_missing_locations(new_tree)

    # Build a mangled name.
    mangled = _mangle_name(template_fn.jit_fn.name, bindings)

    # Import here to avoid circular dependency.
    from .core import JITFunction

    concrete = JITFunction(
        name=mangled,
        fn=template_fn.jit_fn.fn,
        tree=new_tree,
        semantic=template_fn.jit_fn._semantic,
        table=template_fn.jit_fn._table,
        rewrites=template_fn.jit_fn._rewrites,
    )
    # Bind the template values into the new function's globals so the
    # emitter can resolve them during emission.
    concrete._template_bindings = bindings

    template_fn._cache[cache_key] = concrete
    return concrete


# ---------------------------------------------------------------------------
# AST transformer for template substitution.
# ---------------------------------------------------------------------------


class TemplateSubstituter(ast.NodeTransformer):
    """Substitute template parameter names in the AST."""

    def __init__(
        self,
        params: dict[str, TemplateParam],
        bindings: dict[str, Any],
    ):
        self._params = params
        self._bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self._params:
            return node

        param = self._params[node.id]
        value = self._bindings[node.id]

        if param.kind == TemplateParamKind.TYPE:
            # Replace with a sentinel name that the emitter resolves
            # via the symbol table at emission time.
            new_node = ast.Name(id=f"__tpl_type_{node.id}", ctx=node.ctx)
            return ast.copy_location(new_node, node)

        if param.kind == TemplateParamKind.VALUE:
            new_node = ast.Constant(value=value)
            return ast.copy_location(new_node, node)

        if param.kind == TemplateParamKind.CALLABLE:
            new_node = ast.Name(id=f"__tpl_callable_{node.id}", ctx=node.ctx)
            return ast.copy_location(new_node, node)

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        # Mangle the function name with the bindings.
        node.name = _mangle_name(node.name, self._bindings)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        # Substitute type annotations on function arguments.
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        return node


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _validate_bindings(tpl: TemplateFunction, bindings: dict[str, Any]) -> None:
    """Check that all declared params are bound and types are sensible."""
    for name, param in tpl.params.items():
        if name not in bindings:
            raise TypeError(f"template parameter '{name}' not bound in instantiate()")
        value = bindings[name]
        if param.kind == TemplateParamKind.TYPE and not isinstance(value, ASTType):
            raise TypeError(
                f"template parameter '{name}' expects an ASTType, "
                f"got {type(value).__name__}"
            )
        if param.kind == TemplateParamKind.CALLABLE and not callable(value):
            raise TypeError(
                f"template parameter '{name}' expects a callable, "
                f"got {type(value).__name__}"
            )
    extra = set(bindings) - set(tpl.params)
    if extra:
        raise TypeError(f"unexpected template parameters: {', '.join(sorted(extra))}")


def _make_cache_key(
    bindings: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Build a hashable cache key from the bindings."""
    items: list[tuple[str, Any]] = []
    for k in sorted(bindings):
        v = bindings[k]
        # ASTType subclasses are frozen dataclasses, so they're hashable.
        # Callables use identity.
        items.append((k, v))
    return tuple(items)


def _mangle_name(base: str, bindings: dict[str, Any]) -> str:
    """Produce a mangled function name encoding the bindings."""
    parts = [base]
    for k in sorted(bindings):
        v = bindings[k]
        if isinstance(v, ASTType):
            parts.append(str(v).replace("(", "").replace(")", "").replace(" ", ""))
        elif callable(v):
            parts.append(getattr(v, "__name__", "fn"))
        else:
            parts.append(str(v))
    return "_".join(parts)
