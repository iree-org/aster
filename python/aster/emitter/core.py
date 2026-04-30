# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""AST emitter visitor, ``@jit`` decorator, and ``JITFunction``."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable, Optional

from aster import ir

from .arith_emitter import register_arith_emitters
from .context import EmitterContext, get_context_class
from .scf_emitter import register_scf_emitters
from .escape import (
    _REFLECT_SENTINEL,
    execute_escape,
    execute_reflect,
    is_escaped,
)
from .func_emitter import register_func_emitters
from .rewrite import RewritePipeline, get_global_pipeline
from .scope import ScopeStack, SymbolTable
from .semantic import DefaultSemantic, SemanticAnalyzer
from .table import EmitterTable
from .types import ASTType, ASTValue, ScalarType


# ---------------------------------------------------------------------------
# Default table factory.
# ---------------------------------------------------------------------------


def default_table() -> EmitterTable:
    """Return an ``EmitterTable`` pre-populated with func + arith emitters."""
    table = EmitterTable()
    register_func_emitters(table)
    register_arith_emitters(table)
    register_scf_emitters(table)
    return table


# ---------------------------------------------------------------------------
# JITFunction -- the object returned by ``@jit``.
# ---------------------------------------------------------------------------


class JITFunction:
    """A captured Python function ready for AST emission."""

    def __init__(
        self,
        name: str,
        fn: Optional[Callable],
        tree: ast.AST,
        semantic: Optional[SemanticAnalyzer] = None,
        table: Optional[EmitterTable] = None,
        rewrites: Optional[list] = None,
    ):
        self.name = name
        self.fn = fn
        self.tree = tree
        self._semantic = semantic
        self._table = table
        self._rewrites = rewrites or []
        self._template_bindings: dict[str, Any] = {}

    def emit(self, arg_types: Optional[list[ASTType]] = None) -> ir.Module:
        """Parse, rewrite, and emit the function, returning an ``ir.Module``."""
        tree = self.tree

        # Run the AST rewrite pipeline (global + per-jit).
        pipeline = get_global_pipeline()
        tree = pipeline.run(tree)
        if self._rewrites:
            local_pipeline = RewritePipeline()
            for rw in self._rewrites:
                local_pipeline.register(rw)
            tree = local_pipeline.run(tree)

        semantic = self._semantic or DefaultSemantic()
        table = self._table or default_table()

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        loc = ir.Location.unknown(ctx)

        with ctx:
            module = ir.Module.create(loc)
            symbols = SymbolTable()

            # Inject only names the function actually references from its globals
            # (via co_names) rather than the entire module namespace, to keep
            # the symbol table small and predictable.
            if self.fn is not None:
                referenced = set(self.fn.__code__.co_names)
                for k, v in (self.fn.__globals__ or {}).items():
                    if k in referenced:
                        symbols.define(k, v)
                # Closure cells.
                if self.fn.__code__.co_freevars and self.fn.__closure__:
                    for name, cell in zip(
                        self.fn.__code__.co_freevars, self.fn.__closure__
                    ):
                        try:
                            symbols.define(name, cell.cell_contents)
                        except ValueError:
                            pass

            # Inject template bindings into the symbol table.
            for k, v in self._template_bindings.items():
                if isinstance(v, ASTType):
                    symbols.define(f"__tpl_type_{k}", v)
                elif callable(v):
                    symbols.define(f"__tpl_callable_{k}", v)
                else:
                    symbols.define(k, v)

            scope_stack = ScopeStack(symbols)
            ectx = EmitterContext(
                module=module,
                ctx=ctx,
                ip=ir.InsertionPoint(module.body),
                loc=loc,
                scope_stack=scope_stack,
                semantic=semantic,
                table=table,
            )

            emitter = ASTEmitter(ectx)
            if arg_types is not None:
                emitter.set_arg_types(arg_types)
            emitter.visit(tree)

        return module


# ---------------------------------------------------------------------------
# ``@jit`` decorator.
# ---------------------------------------------------------------------------


def jit(
    fn: Optional[Callable] = None,
    *,
    semantic: Optional[SemanticAnalyzer] = None,
    table: Optional[EmitterTable] = None,
    rewrites: Optional[list] = None,
):
    """Decorator that captures a function for AST emission.

    Usage::

        @aster.jit()
        def kernel(a: i32, b: i32) -> i32:
            return a + b

        module = kernel.emit([i32, i32])
    """

    def wrapper(f: Callable) -> JITFunction:
        try:
            source = textwrap.dedent(inspect.getsource(f))
        except OSError as e:
            raise UnsupportedConstruct(
                f"cannot retrieve source for '{f.__name__}': {e}. "
                "Functions defined in interactive sessions or without source "
                "files on disk cannot be used with @jit."
            ) from e
        tree = ast.parse(source)
        return JITFunction(
            name=f.__name__,
            fn=f,
            tree=tree,
            semantic=semantic,
            table=table,
            rewrites=rewrites,
        )

    if fn is not None:
        return wrapper(fn)
    return wrapper


# ---------------------------------------------------------------------------
# Error type.
# ---------------------------------------------------------------------------


class UnsupportedConstruct(Exception):
    """Raised when the emitter encounters an unsupported AST node."""


# ---------------------------------------------------------------------------
# ASTEmitter -- the visitor.
# ---------------------------------------------------------------------------


class ASTEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit MLIR via the emitter context."""

    def __init__(self, ectx: EmitterContext):
        self._ectx = ectx
        self._arg_types: Optional[list[ASTType]] = None

    def set_arg_types(self, arg_types: list[ASTType]) -> None:
        self._arg_types = arg_types

    # -- helpers --------------------------------------------------------------

    def visit_stmts(self, stmts: list[ast.stmt]) -> None:
        """Visit a list of statements in order."""
        for stmt in stmts:
            self.visit(stmt)

    def _resolve_type_annotation(self, annotation: ast.AST) -> ASTType:
        """Resolve a type annotation to an ``ASTType``."""
        if isinstance(annotation, ast.Name):
            # Check the symbol table for template-injected types.
            name = annotation.id
            try:
                resolved = self._ectx.scope_stack.symbols.lookup(name)
                if isinstance(resolved, ASTType):
                    return resolved
            except KeyError:
                pass
            # Look up in the types module.
            from . import types as _types_mod

            val = getattr(_types_mod, name, None)
            if isinstance(val, ASTType):
                return val
            raise TypeError(f"unknown type annotation: {name}")
        if isinstance(annotation, ast.Attribute):
            raise TypeError("dotted type annotations not yet supported")
        raise TypeError(f"unsupported type annotation: {ast.dump(annotation)}")

    def _resolve_return_types(self, node: ast.FunctionDef) -> list[ASTType]:
        """Extract return type annotations from a function def."""
        ret = node.returns
        if ret is None:
            return []
        if isinstance(ret, ast.Tuple):
            return [self._resolve_type_annotation(elt) for elt in ret.elts]
        return [self._resolve_type_annotation(ret)]

    def _resolve_arg_types(self, node: ast.FunctionDef) -> list[ASTType]:
        """Resolve argument types, using explicit arg_types or annotations."""
        if self._arg_types is not None:
            return self._arg_types
        types: list[ASTType] = []
        for arg in node.args.args:
            if arg.annotation is None:
                raise TypeError(
                    f"argument '{arg.arg}' has no type annotation and "
                    "no explicit arg_types were provided"
                )
            types.append(self._resolve_type_annotation(arg.annotation))
        return types

    def _materialize(self, value: Any, target_type: ASTType) -> ASTValue:
        """Materialise a Python constant to an ``ASTValue``."""
        if isinstance(value, ASTValue):
            return value
        if isinstance(value, (int, float)):
            return self._ectx.semantic.materialize_constant(
                self._ectx, value, target_type
            )
        raise TypeError(f"cannot materialise {type(value).__name__} to IR")

    def _ensure_ast_value(
        self, value: Any, peer: Optional[ASTValue] = None
    ) -> ASTValue:
        """Convert constexprs to ``ASTValue`` when flowing into IR ops.

        If *peer* is given, its type is used as the materialisation
        target.
        """
        if isinstance(value, ASTValue):
            return value
        if isinstance(value, (int, float)):
            if peer is not None:
                return self._materialize(value, peer.ast_type)
            # Default: i64 for int, f64 for float.
            if isinstance(value, float):
                target = ScalarType(width=64, is_float=True)
            else:
                target = ScalarType(width=64)
            return self._materialize(value, target)
        raise TypeError(f"expected IR value or constant, got {type(value).__name__}")

    # -- visitor methods ------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Collect emission context decorators (e.g. @aster.my_ctx()).
        ctx_decorators = [
            deco
            for deco in node.decorator_list
            if self._get_context_decorator_name(deco) is not None
        ]
        if len(ctx_decorators) > 1:
            raise UnsupportedConstruct(
                "multiple emission context decorators on the same function are not "
                "supported; nest separate inner functions instead"
            )
        if ctx_decorators:
            ctx_name = self._get_context_decorator_name(ctx_decorators[0])
            self._visit_context_block(ctx_name, node)
            return

        arg_types = self._resolve_arg_types(node)
        return_types = self._resolve_return_types(node)

        saved_return = self._ectx.return_types
        self._ectx.return_types = return_types

        self._ectx.scope_stack.push_scope()
        try:
            fn_emitter = self._ectx.table.get_function_emitter()
            fn_emitter(self._ectx, node, arg_types, self)
        finally:
            self._ectx.scope_stack.pop_scope()
            self._ectx.return_types = saved_return

    def visit_Return(self, node: ast.Return) -> None:
        values: list[ASTValue] = []
        if node.value is not None:
            val = self.visit(node.value)
            if isinstance(val, tuple):
                for v in val:
                    values.append(self._ensure_ast_value(v))
            else:
                values.append(self._ensure_ast_value(val))

        # Validate the return count then coerce each value to its declared type.
        ret_types = self._ectx.return_types
        if len(values) != len(ret_types):
            raise TypeError(
                f"function returns {len(values)} value(s) but declares "
                f"{len(ret_types)} return type(s)"
            )
        if ret_types:
            values = [
                self._ectx.semantic.coerce(self._ectx, v, rt)
                for v, rt in zip(values, ret_types)
            ]

        ret_emitter = self._ectx.table.get_return_emitter()
        ret_emitter(self._ectx, values)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._ectx.scope_stack.bind_local(target.id, value)
            elif isinstance(target, ast.Tuple):
                if not isinstance(value, (tuple, list)):
                    raise UnsupportedConstruct("tuple unpacking requires a tuple RHS")
                for elt, v in zip(target.elts, value):
                    if isinstance(elt, ast.Name):
                        self._ectx.scope_stack.bind_local(elt.id, v)
                    else:
                        raise UnsupportedConstruct(
                            f"unsupported assignment target: {ast.dump(elt)}"
                        )
            else:
                raise UnsupportedConstruct(
                    f"unsupported assignment target: {ast.dump(target)}"
                )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # Desugar ``x += y`` into ``x = x + y``.  Read the current value of
        # the target directly via name resolution rather than dispatching
        # through visit() to avoid ambiguity with Store-context Name nodes.
        if not isinstance(node.target, ast.Name):
            raise UnsupportedConstruct(
                f"augmented assignment target must be a simple name, "
                f"got {ast.dump(node.target)}"
            )
        try:
            lhs = self._ectx.scope_stack.resolve(node.target.id)
        except KeyError:
            raise NameError(
                f"name '{node.target.id}' is not defined before augmented assignment"
            ) from None
        rhs = self.visit(node.value)
        lhs = self._ensure_ast_value(lhs)
        rhs = self._ensure_ast_value(rhs, peer=lhs)
        binop_emitter = self._ectx.table.get_binop_emitter(type(node.op))
        result = binop_emitter(self._ectx, lhs, rhs)
        self._ectx.scope_stack.bind_local(node.target.id, result)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        cond = self.visit(node.test)
        cond = self._ensure_ast_value(cond)
        if_emitter = self._ectx.table.get_if_emitter()
        if_emitter(self._ectx, cond, node.body, node.orelse, self)

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise UnsupportedConstruct(
                "only simple loop variables are supported (e.g. `for i in ...`)"
            )
        target_name = node.target.id

        # Expect `range(lb, ub)` or `range(lb, ub, step)` or `range(ub)`.
        if not isinstance(node.iter, ast.Call):
            raise UnsupportedConstruct("for-loop iterator must be a range() call")
        iter_func = node.iter.func
        if not (isinstance(iter_func, ast.Name) and iter_func.id == "range"):
            raise UnsupportedConstruct("for-loop iterator must be a range() call")

        range_args = [self.visit(a) for a in node.iter.args]
        from .types import IndexType

        idx_type = IndexType()
        if len(range_args) == 1:
            lb = self._materialize(0, idx_type)
            ub = self._ensure_ast_value(range_args[0])
            step = self._materialize(1, idx_type)
        elif len(range_args) == 2:
            lb = self._ensure_ast_value(range_args[0])
            ub = self._ensure_ast_value(range_args[1])
            step = self._materialize(1, idx_type)
        elif len(range_args) == 3:
            lb = self._ensure_ast_value(range_args[0])
            ub = self._ensure_ast_value(range_args[1])
            step = self._ensure_ast_value(range_args[2])
        else:
            raise UnsupportedConstruct("range() expects 1-3 arguments")

        for_emitter = self._ectx.table.get_for_emitter()
        for_emitter(self._ectx, target_name, lb, ub, step, node.body, self)

    # -- expressions ----------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> ASTValue:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        # Materialise constants using peer type.
        if isinstance(lhs, ASTValue) and not isinstance(rhs, ASTValue):
            rhs = self._ensure_ast_value(rhs, peer=lhs)
        elif isinstance(rhs, ASTValue) and not isinstance(lhs, ASTValue):
            lhs = self._ensure_ast_value(lhs, peer=rhs)
        else:
            lhs = self._ensure_ast_value(lhs)
            rhs = self._ensure_ast_value(rhs)
        emitter = self._ectx.table.get_binop_emitter(type(node.op))
        return emitter(self._ectx, lhs, rhs)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ASTValue:
        operand = self.visit(node.operand)
        operand = self._ensure_ast_value(operand)
        emitter = self._ectx.table.get_unaryop_emitter(type(node.op))
        return emitter(self._ectx, operand)

    def visit_Compare(self, node: ast.Compare) -> ASTValue:
        # Only single comparisons for now.
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsupportedConstruct("chained comparisons not supported")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        if isinstance(lhs, ASTValue) and not isinstance(rhs, ASTValue):
            rhs = self._ensure_ast_value(rhs, peer=lhs)
        elif isinstance(rhs, ASTValue) and not isinstance(lhs, ASTValue):
            lhs = self._ensure_ast_value(lhs, peer=rhs)
        else:
            lhs = self._ensure_ast_value(lhs)
            rhs = self._ensure_ast_value(rhs)
        emitter = self._ectx.table.get_cmpop_emitter(type(node.ops[0]))
        return emitter(self._ectx, lhs, rhs)

    def visit_Constant(self, node: ast.Constant) -> Any:
        # Return the raw Python value; materialised lazily.
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        try:
            return self._ectx.scope_stack.resolve(node.id)
        except KeyError:
            raise NameError(f"name '{node.id}' is not defined") from None

    def visit_Call(self, node: ast.Call) -> Any:
        callee = self.visit(node.func)

        # Handle aster.reflect(fn, *args).
        if callee is _REFLECT_SENTINEL:
            if not node.args:
                raise UnsupportedConstruct("reflect() requires at least one argument")
            fn = self.visit(node.args[0])
            resolved_args = [self.visit(a) for a in node.args[1:]]
            result = execute_reflect(self._ectx, fn, resolved_args)
            return result

        # Handle @escape-decorated functions.
        if is_escaped(callee):
            resolved_args = [self.visit(a) for a in node.args]
            return execute_escape(self._ectx, callee, resolved_args)

        # Handle aster.instantiate().
        from .template import instantiate as _instantiate

        if callee is _instantiate:
            # aster.instantiate(template_fn, **kwargs) -- evaluate args.
            if not node.args:
                raise UnsupportedConstruct(
                    "instantiate() requires the template function as first arg"
                )
            tpl_fn = self.visit(node.args[0])
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = self.visit(kw.value)
            concrete = _instantiate(tpl_fn, **kwargs)
            return concrete

        # Check the emitter table for a registered call emitter.
        call_emitter = self._ectx.table.get_call_emitter(callee)
        if call_emitter is not None:
            resolved_args = [self.visit(a) for a in node.args]
            ast_args = [self._ensure_ast_value(a) for a in resolved_args]
            return call_emitter(self._ectx, callee, ast_args)

        raise UnsupportedConstruct(f"unsupported call to {callee!r}")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Resolve dotted names (e.g., aster.reflect).
        value = self.visit(node.value)
        return getattr(value, node.attr)

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self.visit(elt) for elt in node.elts)

    # -- emission context blocks ----------------------------------------------

    def _get_context_decorator_name(self, deco: ast.AST) -> Optional[str]:
        """If *deco* is ``@aster.xxx()`` and xxx is a registered context, return xxx."""
        if isinstance(deco, ast.Call):
            deco = deco.func
        if isinstance(deco, ast.Attribute):
            ctx_cls = get_context_class(deco.attr)
            if ctx_cls is not None:
                return deco.attr
        if isinstance(deco, ast.Name):
            ctx_cls = get_context_class(deco.id)
            if ctx_cls is not None:
                return deco.id
        return None

    def _visit_context_block(self, ctx_name: str, node: ast.FunctionDef) -> None:
        """Handle a nested ``@aster.ctx_name()`` block."""
        ctx_cls = get_context_class(ctx_name)
        assert ctx_cls is not None
        emission_ctx = ctx_cls()

        # Save state.
        saved_table = self._ectx.table
        saved_semantic = self._ectx.semantic
        saved_symbols = self._ectx.scope_stack.symbols

        # Configure new table and semantic.
        new_table = emission_ctx.configure_table(saved_table)
        new_semantic = emission_ctx.configure_semantic(saved_semantic)
        scope_config = emission_ctx.configure_scope()

        # Fork symbol table.
        self._ectx.scope_stack.symbols = saved_symbols.fork()
        self._ectx.table = new_table
        self._ectx.semantic = new_semantic

        # Push scope.
        if scope_config.get("isolated", False):
            self._ectx.scope_stack.push_isolated_scope()
        else:
            self._ectx.scope_stack.push_scope()

        try:
            self.visit_stmts(node.body)
        finally:
            self._ectx.scope_stack.pop_scope()
            self._ectx.scope_stack.symbols = saved_symbols
            self._ectx.table = saved_table
            self._ectx.semantic = saved_semantic

    # -- fallback -------------------------------------------------------------

    def generic_visit(self, node: ast.AST) -> Any:
        raise UnsupportedConstruct(f"unsupported AST construct: {type(node).__name__}")
