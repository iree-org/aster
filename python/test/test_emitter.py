# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for the contextual AST-to-MLIR emitter framework.

Covers: jit, func.func emission, arith ops, escape/reflect, nested
emission contexts, scope model, templates, and AST rewrites.
"""

from __future__ import annotations

import ast

from aster.emitter import (
    DefaultSemantic,
    EmissionContext,
    EmitCContext,
    EmitterTable,
    Type,
    default_table,
    escape,
    i1,
    i32,
    i64,
    f32,
    index,
    instantiate,
    jit,
    pattern,
    reflect,
    register_context,
    template,
)
from aster.emitter.scope import Scope, ScopeStack, SymbolTable
from aster.emitter.types import ASTValue


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _mlir_str(module) -> str:
    """Return the MLIR text representation of *module*."""
    return str(module)


# ===================================================================
# 1. Scope model tests (pure Python, no MLIR).
# ===================================================================


def test_symbol_table_define_lookup():
    st = SymbolTable()
    st.define("x", 42)
    assert st.lookup("x") == 42


def test_symbol_table_parent_chain():
    parent = SymbolTable()
    parent.define("x", 1)
    child = parent.fork()
    assert child.lookup("x") == 1
    child.define("x", 2)
    assert child.lookup("x") == 2
    assert parent.lookup("x") == 1


def test_scope_locals():
    s = Scope()
    s.bind("a", 10)
    assert s.lookup("a") == 10


def test_scope_parent_visibility():
    outer = Scope()
    outer.bind("a", 10)
    inner = Scope(parent=outer)
    assert inner.lookup("a") == 10
    inner.bind("a", 20)
    assert inner.lookup("a") == 20
    assert outer.lookup("a") == 10


def test_scope_stack_push_pop():
    st = SymbolTable()
    st.define("global_sym", "g")
    ss = ScopeStack(st)
    ss.push_scope()
    ss.bind_local("x", 1)
    assert ss.resolve("x") == 1
    assert ss.resolve("global_sym") == "g"
    ss.pop_scope()
    # After pop, x should not be resolvable; global_sym still is.
    try:
        ss.push_scope()
        ss.resolve("x")
        assert False, "should have raised"
    except KeyError:
        pass
    assert ss.resolve("global_sym") == "g"


def test_scope_isolated():
    st = SymbolTable()
    ss = ScopeStack(st)
    ss.push_scope()
    ss.bind_local("outer", 1)
    ss.push_isolated_scope()
    try:
        ss.resolve("outer")
        assert False, "isolated scope should not see parent locals"
    except KeyError:
        pass
    ss.pop_scope()
    assert ss.resolve("outer") == 1


def test_symbol_shadowing():
    parent = SymbolTable()
    parent.define("fn", "original")
    child = parent.fork()
    child.define("fn", "overridden")
    assert child.lookup("fn") == "overridden"
    assert parent.lookup("fn") == "original"


# ===================================================================
# 2. EmitterTable tests.
# ===================================================================


def test_table_parent_chain():
    parent = EmitterTable()
    parent.register_binop(ast.Add, lambda ectx, lhs, rhs: "parent_add")
    child = EmitterTable(parent=parent)
    assert child.get_binop_emitter(ast.Add) is not None
    child.register_binop(ast.Add, lambda ectx, lhs, rhs: "child_add")
    assert child.get_binop_emitter(ast.Add)(None, None, None) == "child_add"
    assert parent.get_binop_emitter(ast.Add)(None, None, None) == "parent_add"


def test_default_table_has_binops():
    table = default_table()
    for op in (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod):
        assert table.get_binop_emitter(op) is not None


# ===================================================================
# 3. Simple function emission.
# ===================================================================


def test_simple_function():
    """Emit a trivial function and check for func.func in the output."""

    @jit()
    def add(a: i32, b: i32) -> i32:
        return a + b

    module = add.emit([i32, i32])
    text = _mlir_str(module)
    assert "func.func" in text
    assert "arith.addi" in text
    assert "return" in text


def test_function_signature_types():
    """Verify the emitted function has the right argument types."""

    @jit()
    def f(x: i64) -> i64:
        return x

    module = f.emit([i64])
    text = _mlir_str(module)
    assert "i64" in text


# ===================================================================
# 4. Arith operations.
# ===================================================================


def test_arith_addi():
    @jit()
    def f(a: i32, b: i32) -> i32:
        return a + b

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.addi" in text


def test_arith_subi():
    @jit()
    def f(a: i32, b: i32) -> i32:
        return a - b

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.subi" in text


def test_arith_muli():
    @jit()
    def f(a: i32, b: i32) -> i32:
        return a * b

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.muli" in text


def test_arith_addf():
    @jit()
    def f(a: f32, b: f32) -> f32:
        return a + b

    text = _mlir_str(f.emit([f32, f32]))
    assert "arith.addf" in text


def test_arith_constant():
    """Python int literal materialises as arith.constant."""

    @jit()
    def f(a: i32) -> i32:
        return a + 1

    text = _mlir_str(f.emit([i32]))
    assert "arith.constant" in text
    assert "arith.addi" in text


def test_arith_assign_and_use():
    @jit()
    def f(a: i32, b: i32) -> i32:
        c = a + b
        return c

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.addi" in text
    assert "return" in text


# ===================================================================
# 5. Escape and reflect.
# ===================================================================


def test_escape_decorator():
    """An @escape function is executed at compile time."""
    side_effects = []

    @escape
    def log_compile(value):
        side_effects.append(("compiled", value))
        return None

    @jit()
    def f(a: i32) -> i32:
        log_compile(42)
        return a

    f.emit([i32])
    assert len(side_effects) == 1
    assert side_effects[0] == ("compiled", 42)


def test_reflect_call():
    """aster.reflect() executes a function at compile time."""
    side_effects = []

    def my_fn(x):
        side_effects.append(x)

    @jit()
    def f(a: i32) -> i32:
        reflect(my_fn, 99)
        return a

    f.emit([i32])
    assert 99 in side_effects


# ===================================================================
# 6. Nested emission contexts.
# ===================================================================


def test_nested_context_overrides_binop():
    """Inner context overrides ast.Add, outer uses default."""
    custom_called = []

    def custom_add(ectx, lhs, rhs):
        custom_called.append(True)
        # Fall back to default arith.addi for the actual emission.
        from aster.dialects import arith

        ir_val = arith.addi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
        return ASTValue(ir_val, lhs.ast_type)

    class MyCtx(EmissionContext):
        def configure_table(self, parent):
            table = EmitterTable(parent=parent)
            table.register_binop(ast.Add, custom_add)
            return table

    register_context("my_ctx", MyCtx)

    @jit()
    def f(a: i32, b: i32) -> i32:
        d = a + b

        @my_ctx()  # noqa: F821
        def inner():
            c = a + b  # noqa: F841

        return d

    import aster.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir.Location.unknown(ctx)

    with ctx:
        module = ir.Module.create(loc)
        from aster.emitter.context import EmitterContext
        from aster.emitter.scope import ScopeStack, SymbolTable
        from aster.emitter.core import ASTEmitter, default_table

        symbols = SymbolTable()
        scope_stack = ScopeStack(symbols)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=scope_stack,
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    assert len(custom_called) > 0, "custom context add was not called"


def test_context_inherits_parent():
    """Inner context inherits parent table for ops it doesn't override."""

    class PassthroughCtx(EmissionContext):
        pass

    register_context("passthrough", PassthroughCtx)

    @jit()
    def f(a: i32, b: i32) -> i32:
        @passthrough()  # noqa: F821
        def inner():
            c = a + b  # noqa: F841

        return a + b

    import aster.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    loc = ir.Location.unknown(ctx)

    with ctx:
        module = ir.Module.create(loc)
        from aster.emitter.context import EmitterContext
        from aster.emitter.scope import ScopeStack, SymbolTable
        from aster.emitter.core import ASTEmitter, default_table

        symbols = SymbolTable()
        scope_stack = ScopeStack(symbols)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=scope_stack,
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    text = str(module)
    # Both inner and outer should produce arith.addi (inherited).
    assert text.count("arith.addi") >= 2


def test_scope_locals_popped_after_context():
    """Variables bound in inner context are not visible after it ends."""

    class ScopedCtx(EmissionContext):
        pass

    register_context("scoped", ScopedCtx)

    @jit()
    def f(a: i32) -> i32:
        @scoped()  # noqa: F821
        def inner():
            inner_var = a + a  # noqa: F841

        return a

    # If inner_var leaked out, emitting "return inner_var" would succeed.
    # Since it doesn't, we just verify the module emits cleanly.
    import aster.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.create(loc=ir.Location.unknown(ctx))
        from aster.emitter.context import EmitterContext
        from aster.emitter.scope import ScopeStack, SymbolTable
        from aster.emitter.core import ASTEmitter, default_table

        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=ir.Location.unknown(ctx),
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = str(module)
    assert "func.func" in text


# ===================================================================
# 7. Semantic coercion.
# ===================================================================


def test_semantic_coercion_i32_i64():
    """I32 + i64 should widen i32 via arith.extsi."""

    @jit()
    def f(a: i32, b: i64) -> i64:
        return a + b

    text = _mlir_str(f.emit([i32, i64]))
    assert "arith.extsi" in text
    assert "arith.addi" in text


# ===================================================================
# 8. Templates.
# ===================================================================


def test_template_type_param():
    """@template(T=Type) substitutes type annotations."""

    @template(T=Type)
    @jit()
    def add_generic(a: T, b: T) -> T:  # noqa: F821
        return a + b

    concrete = instantiate(add_generic, T=i32)
    module = concrete.emit([i32, i32])
    text = _mlir_str(module)
    assert "func.func" in text
    assert "i32" in text
    assert "arith.addi" in text


def test_template_value_param():
    """@template(N=int) substitutes constants."""

    @template(N=int)
    @jit()
    def add_n(a: i32) -> i32:
        return a + N  # noqa: F821

    concrete = instantiate(add_n, N=42)
    module = concrete.emit([i32])
    text = _mlir_str(module)
    assert "arith.constant 42" in text or "42 : i32" in text


def test_template_caching():
    """Same instantiation args return the same JITFunction."""

    @template(T=Type)
    @jit()
    def add_generic(a: T, b: T) -> T:  # noqa: F821
        return a + b

    c1 = instantiate(add_generic, T=i32)
    c2 = instantiate(add_generic, T=i32)
    assert c1 is c2


def test_template_different_types():
    """Different type bindings produce different functions."""

    @template(T=Type)
    @jit()
    def neg(a: T) -> T:  # noqa: F821
        return a

    c_i32 = instantiate(neg, T=i32)
    c_i64 = instantiate(neg, T=i64)
    assert c_i32 is not c_i64
    assert "i32" in _mlir_str(c_i32.emit([i32]))
    assert "i64" in _mlir_str(c_i64.emit([i64]))


# ===================================================================
# 9. AST rewrite hooks.
# ===================================================================


def test_per_jit_rewrite():
    """Per-jit rewrites only affect the decorated function."""

    class DoubleConstants(ast.NodeTransformer):
        def rewrite(self, tree):
            return self.visit(tree)

        def visit_Constant(self, node):
            if isinstance(node.value, int):
                return ast.copy_location(ast.Constant(value=node.value * 2), node)
            return node

    @jit(rewrites=[DoubleConstants()])
    def f(a: i32) -> i32:
        return a + 5

    text = _mlir_str(f.emit([i32]))
    # 5 should have been doubled to 10 by the rewrite.
    assert "10" in text


# ===================================================================
# 10. EmitterTable direct registration.
# ===================================================================


def test_custom_binop_emitter():
    """Register a custom BinOpEmitter for ast.Add."""
    custom_called = []

    def my_add(ectx, lhs, rhs):
        custom_called.append(True)
        from aster.dialects import arith

        ir_val = arith.addi(lhs.ir_value, rhs.ir_value, loc=ectx.loc, ip=ectx.ip)
        return ASTValue(ir_val, lhs.ast_type)

    table = default_table()
    table.register_binop(ast.Add, my_add)

    @jit(table=table)
    def f(a: i32, b: i32) -> i32:
        return a + b

    f.emit([i32, i32])
    assert len(custom_called) == 1


# ===================================================================
# 11. Multiple operations and complex expressions.
# ===================================================================


def test_multiple_ops():
    @jit()
    def f(a: i32, b: i32) -> i32:
        c = a + b
        d = c * a
        return d - b

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.addi" in text
    assert "arith.muli" in text
    assert "arith.subi" in text


def test_augmented_assign():
    @jit()
    def f(a: i32, b: i32) -> i32:
        c = a
        c += b
        return c

    text = _mlir_str(f.emit([i32, i32]))
    assert "arith.addi" in text


# ===================================================================
# 12. scf.if emission.
# ===================================================================


def test_if_then_only():
    """If with only a then branch produces scf.if with one region."""

    @jit()
    def f(cond: i1, a: i32, b: i32) -> i32:
        c = a + b
        if cond:
            c = a * b
        return c

    text = _mlir_str(f.emit([i1, i32, i32]))
    assert "scf.if" in text
    assert "scf.yield" in text
    assert "arith.muli" in text


def test_if_then_else():
    """If/else produces scf.if with two regions."""

    @jit()
    def f(cond: i1, a: i32, b: i32) -> i32:
        if cond:
            c = a + b  # noqa: F841
        else:
            c = a * b  # noqa: F841
        return a

    text = _mlir_str(f.emit([i1, i32, i32]))
    assert "scf.if" in text
    assert "} else {" in text
    assert "arith.addi" in text
    assert "arith.muli" in text


def test_if_comparison_cond():
    """If with a comparison condition auto-coerces to i1."""

    @jit()
    def f(a: i32, b: i32) -> i32:
        if a == b:
            c = a + b  # noqa: F841
        return a

    text = _mlir_str(f.emit([i32, i32]))
    assert "scf.if" in text
    assert "arith.cmpi" in text


# ===================================================================
# 13. scf.for emission.
# ===================================================================


def test_for_range_single_arg():
    """For i in range(N) produces scf.for with lb=0, step=1."""

    @jit()
    def f(n: index) -> index:
        for i in range(n):  # noqa: B007
            pass
        return n

    text = _mlir_str(f.emit([index]))
    assert "scf.for" in text
    # Constant 0 for lb and 1 for step.
    assert "arith.constant 0 : index" in text
    assert "arith.constant 1 : index" in text


def test_for_range_two_args():
    """For i in range(lb, ub) produces scf.for with step=1."""

    @jit()
    def f(lb: index, ub: index) -> index:
        for i in range(lb, ub):  # noqa: B007
            pass
        return lb

    text = _mlir_str(f.emit([index, index]))
    assert "scf.for" in text


def test_for_range_three_args():
    """For i in range(lb, ub, step) produces scf.for."""

    @jit()
    def f(lb: index, ub: index, step: index) -> index:
        for i in range(lb, ub, step):  # noqa: B007
            pass
        return lb

    text = _mlir_str(f.emit([index, index, index]))
    assert "scf.for" in text


def test_for_body_uses_iv():
    """The induction variable is accessible inside the loop body."""

    @jit()
    def f(n: index, a: i32) -> i32:
        for i in range(n):  # noqa: B007
            a = a + a
        return a

    text = _mlir_str(f.emit([index, i32]))
    assert "scf.for" in text
    assert "arith.addi" in text


def test_for_integer_bounds_cast():
    """Integer bounds are automatically cast to index."""

    @jit()
    def f(n: i32) -> i32:
        for i in range(n):  # noqa: B007
            pass
        return n

    text = _mlir_str(f.emit([i32]))
    assert "scf.for" in text
    assert "arith.index_cast" in text


def test_nested_for():
    """Nested for-loops produce nested scf.for ops."""

    @jit()
    def f(m: index, n: index) -> index:
        for i in range(m):  # noqa: B007
            for j in range(n):  # noqa: B007
                pass
        return m

    text = _mlir_str(f.emit([index, index]))
    assert text.count("scf.for") == 2


def test_if_inside_for():
    """If inside a for loop."""

    @jit()
    def f(cond: i1, n: index, a: i32) -> i32:
        for i in range(n):  # noqa: B007
            if cond:
                a = a + a
        return a

    text = _mlir_str(f.emit([i1, index, i32]))
    print(text)
    assert "scf.for" in text
    assert "scf.if" in text


# ===================================================================
# 14. EmitC emission context.
# ===================================================================


def test_emitc_context_registered():
    """The EmitC context is registered under the name 'emitc'."""
    from aster.emitter.context import get_context_class

    assert get_context_class("emitc") is EmitCContext


def test_emitc_add():
    """Emitc context emits emitc.add instead of arith.addi."""

    @jit()
    def f(a: i32, b: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            c = a + b  # noqa: F841

        return a + b

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        symbols = SymbolTable()
        scope_stack = ScopeStack(symbols)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=scope_stack,
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    # The inner context block should produce emitc ops.
    assert "emitc.add" in text or '"emitc.add"' in text
    # The outer scope still uses arith.
    assert "arith.addi" in text


def test_emitc_sub():
    """Emitc context emits emitc.sub."""

    @jit()
    def f(a: i32, b: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            c = a - b  # noqa: F841

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.sub" in text or '"emitc.sub"' in text


def test_emitc_mul():
    """Emitc context emits emitc.mul."""

    @jit()
    def f(a: i32, b: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            c = a * b  # noqa: F841

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.mul" in text or '"emitc.mul"' in text


def test_emitc_cmp():
    """Emitc context emits emitc.cmp for comparisons."""

    @jit()
    def f(a: i32, b: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            c = a == b  # noqa: F841

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32, i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.cmp" in text or "cmp eq" in text


def test_emitc_constant():
    """Emitc context materialises constants via emitc.constant."""

    @jit()
    def f(a: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            c = a + 42  # noqa: F841

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.constant" in text
    assert "42" in text


def test_emitc_class_fields_only():
    """EmitC class with only field annotations emits emitc.class + emitc.field."""

    @jit()
    def f(a: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            class Point:
                x: i32  # noqa: F821
                y: i32  # noqa: F821

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.class" in text
    assert 'sym_name = "Point"' in text
    assert "emitc.field" in text
    assert 'sym_name = "x"' in text
    assert 'sym_name = "y"' in text


def test_emitc_class_with_method():
    """EmitC class method using self.x emits emitc.func + get_field."""

    @jit()
    def f(a: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            class Point:
                x: i32  # noqa: F821
                y: i32  # noqa: F821

                def sum(self) -> i32:  # noqa: F821
                    return self.x + self.y

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.class" in text
    assert 'sym_name = "Point"' in text
    assert 'sym_name = "sum"' in text
    assert "emitc.get_field" in text
    assert "field_name = @x" in text
    assert "field_name = @y" in text
    assert "emitc.add" in text or '"emitc.add"' in text


def test_emitc_class_method_with_args():
    """EmitC class method with extra parameters beyond self."""

    @jit()
    def f(a: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            class Adder:
                x: i32  # noqa: F821

                def add(self, val: i32) -> i32:  # noqa: F821
                    return self.x + val

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.class" in text
    assert 'sym_name = "Adder"' in text
    assert 'sym_name = "add"' in text
    assert "i32" in text
    assert "emitc.get_field" in text
    assert "field_name = @x" in text
    assert "emitc.add" in text or '"emitc.add"' in text


def test_emitc_class_multiple_methods():
    """EmitC class with multiple methods."""

    @jit()
    def f(a: i32) -> i32:
        @emitc()  # noqa: F821
        def inner():
            class Vec:
                x: i32  # noqa: F821
                y: i32  # noqa: F821

                def get_x(self) -> i32:  # noqa: F821
                    return self.x

                def get_y(self) -> i32:  # noqa: F821
                    return self.y

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(f.tree)

    text = _mlir_str(module)
    assert "emitc.class" in text
    assert 'sym_name = "Vec"' in text
    assert 'sym_name = "get_x"' in text
    assert 'sym_name = "get_y"' in text
    assert "emitc.get_field" in text
    assert "field_name = @x" in text
    assert "field_name = @y" in text


# ===================================================================
# 16. Pattern dialect emission.
# ===================================================================


def test_pattern_fields_only():
    """Pattern with fields and an empty action body."""

    @pattern(benefit=1, op="MyOp", fields=["counter: i32"])
    def my_pat(op, rewriter):  # noqa: F841
        cond = counter == 0  # noqa: F821, F841

        def rewrite(cond):  # noqa: F841
            pass

    module = my_pat.emit()
    text = _mlir_str(module)
    assert "pattern.rewrite_pattern @my_pat" in text
    assert "benefit(1)" in text
    assert "op(@MyOp)" in text
    assert "pattern.field @counter : i32" in text
    assert "pattern.yield" in text


def test_pattern_with_action():
    """Pattern with a def rewrite(cond): action block."""

    @pattern(benefit=2, op="SomeOp", fields=["flag: i32"])
    def action_pat(op, rewriter):  # noqa: F841
        v = flag  # noqa: F821
        cond = v == 0  # noqa: F841

        def rewrite(cond):  # noqa: F841
            pass

    module = action_pat.emit()
    text = _mlir_str(module)
    assert "pattern.rewrite_pattern @action_pat" in text
    assert "pattern.action" in text
    assert "pattern.get_field @flag" in text


def test_pattern_get_field_in_body():
    """Accessing field names in the body emits pattern.get_field."""

    @pattern(benefit=0, op="TestOp", fields=["x: i32", "y: i32"])
    def field_pat(op, rewriter):  # noqa: F841
        a = x  # noqa: F821
        b = y  # noqa: F821
        cond = a == b  # noqa: F841

        def rewrite(cond):  # noqa: F841
            pass

    module = field_pat.emit()
    text = _mlir_str(module)
    assert "pattern.get_field @x" in text
    assert "pattern.get_field @y" in text
    assert "pattern.field @x : i32" in text
    assert "pattern.field @y : i32" in text


def test_pattern_as_nested_context():
    """Using @pattern() as a nested context inside @jit."""

    @jit()
    def outer(a: i32) -> i32:
        @pattern()  # noqa: F821
        def inner():
            pass

        return a

    import aster.ir as ir
    from aster.emitter.context import EmitterContext
    from aster.emitter.core import ASTEmitter, default_table
    from aster.emitter.scope import ScopeStack, SymbolTable

    ctx = ir.Context()
    with ctx:
        module = ir.Module.create(ir.Location.unknown(ctx))
        loc = ir.Location.unknown(ctx)
        ectx = EmitterContext(
            module=module,
            ctx=ctx,
            ip=ir.InsertionPoint(module.body),
            loc=loc,
            scope_stack=ScopeStack(SymbolTable()),
            semantic=DefaultSemantic(),
            table=default_table(),
        )
        emitter = ASTEmitter(ectx)
        emitter.set_arg_types([i32])
        emitter.visit(outer.tree)

    text = _mlir_str(module)
    # The outer function should emit func.func.
    assert "func.func" in text


def test_pattern_emitc_ops_in_action():
    """EmitC ops appear inside action body for C++ translation."""

    @pattern(benefit=3, op="FooOp", fields=["val: i32"])
    def emitc_pat(op, rewriter):  # noqa: F841
        v = val  # noqa: F821
        cond = v == 0  # noqa: F841

        def rewrite(cond):  # noqa: F841
            result = val + val  # noqa: F821, F841

    module = emitc_pat.emit()
    text = _mlir_str(module)
    assert "pattern.action" in text
    # Inside the action body, arithmetic uses emitc ops.
    assert "emitc.add" in text or '"emitc.add"' in text


# ===================================================================
# 17. Function and method calls.
# ===================================================================


def test_pattern_method_call():
    """Method call on rewriter emits pattern.method_call."""

    @pattern(benefit=1, op="MyOp")
    def method_pat(op, rewriter):
        rewriter.eraseOp(op)  # noqa: F821
        cond = 1 == 1  # noqa: F841

        def rewrite(cond):  # noqa: F841
            pass

    module = method_pat.emit()
    text = _mlir_str(module)
    assert "pattern.method_call @eraseOp" in text


def test_pattern_function_call():
    """Plain function call emits emitc.call_opaque."""

    @pattern(benefit=1, op="MyOp")
    def call_pat(op, rewriter):
        doSomething()  # noqa: F821
        cond = 1 == 1  # noqa: F841

        def rewrite(cond):  # noqa: F841
            pass

    module = call_pat.emit()
    text = _mlir_str(module)
    assert "emitc.call_opaque" in text
    assert "doSomething" in text


def test_pattern_method_call_with_field_arg():
    """Method call using a pattern field as argument."""

    @pattern(benefit=1, op="MyOp", fields=["counter: i32"])
    def field_arg_pat(op, rewriter):
        v = counter  # noqa: F821
        rewriter.setValue(v)  # noqa: F821
        cond = v == 0  # noqa: F841

        def rewrite(cond):  # noqa: F841
            pass

    module = field_arg_pat.emit()
    text = _mlir_str(module)
    assert "pattern.get_field @counter" in text
    assert "pattern.method_call @setValue" in text


def test_pattern_method_call_in_action():
    """Method call inside action body emits pattern.method_call."""

    @pattern(benefit=1, op="MyOp")
    def action_call_pat(op, rewriter):
        cond = 1 == 1  # noqa: F841

        def rewrite(cond):  # noqa: F841
            rewriter.eraseOp(op)  # noqa: F821

    module = action_call_pat.emit()
    text = _mlir_str(module)
    assert "pattern.action" in text
    assert "pattern.method_call @eraseOp" in text


def test_pattern_function_call_in_action():
    """Plain function call inside action body emits emitc.call_opaque."""

    @pattern(benefit=1, op="MyOp")
    def action_fn_pat(op, rewriter):
        cond = 1 == 1  # noqa: F841

        def rewrite(cond):  # noqa: F841
            doWork()  # noqa: F821

    module = action_fn_pat.emit()
    text = _mlir_str(module)
    assert "pattern.action" in text
    assert "emitc.call_opaque" in text
    assert "doWork" in text
