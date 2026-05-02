# AST-to-MLIR Emitter Framework

A contextual, extensible Python AST-to-MLIR emitter inspired by Triton's
approach.  Python functions decorated with `@jit` are captured as AST trees
and lowered to MLIR through a pluggable pipeline of emitter tables, semantic
analysers, AST rewrites, and emission contexts.

## Architecture overview

```
Python source
     |
     v
  ast.parse        ──>  AST tree
     |
  RewritePipeline  ──>  (optional) AST rewrites / template substitution
     |
  ASTEmitter       ──>  walks the AST, dispatches to EmitterTable handlers
     |                     ├── BinOp/UnaryOp/CmpOp emitters
     |                     ├── FunctionEmitter, ReturnEmitter
     |                     ├── IfEmitter, ForEmitter
     |                     ├── CallEmitter, MethodCallEmitter
     |                     ├── ClassEmitter, AttributeEmitter, NameEmitter
     |                     └── UnresolvedNameHandler
     |
  SemanticAnalyzer ──>  type inference, coercion, constant materialisation
     |
     v
  MLIR ir.Module
```

The emitter table and semantic analyser are swappable per *emission context*.
A nested `@aster.<context>()` decorator inside a `@jit` function switches to
a different table/semantic pair for that scope, allowing the same Python
syntax to produce different MLIR dialects.

## Features

### 1. Core emitter (`@jit`)

Captures a Python function, parses its source into an AST, and emits MLIR.
The default table produces `func.func` operations with `arith` dialect
arithmetic and `scf` dialect control flow.

```python
from aster.emitter import jit, i32

@jit()
def add(a: i32, b: i32) -> i32:
    return a + b

module = add.emit([i32, i32])
# func.func @add(%arg0: i32, %arg1: i32) -> i32 {
#   %0 = arith.addi %arg0, %arg1 : i32
#   return %0 : i32
# }
```

Supported constructs in the default context:

- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `<<`, `>>`, `&`, `|`, `^`
  (maps to `arith.addi`/`addf`, `subi`/`subf`, `muli`/`mulf`, etc.).
- **Unary**: `+x`, `-x`, `~x`, `not x`.
- **Comparisons**: `==`, `!=`, `<`, `<=`, `>`, `>=`
  (maps to `arith.cmpi`/`cmpf`).
- **Constants**: integer and float literals materialised via
  `arith.constant`.
- **Assignments**: `x = expr` and `x += expr` (augmented assignment).
- **If/else**: `if cond: ... else: ...` maps to `scf.if`.
- **For loops**: `for i in range(...)` maps to `scf.for` (1, 2, or 3 arg
  `range`).
- **Return**: `return expr` maps to `func.return`.

### 2. Type system

Frontend types (`ASTType` subclasses) describe the types flowing through the
emitter and know how to produce their MLIR counterparts.

| Python alias | MLIR type |
|---|---|
| `i1` | `i1` |
| `i8` | `i8` |
| `i16` | `i16` |
| `i32` | `i32` |
| `i64` | `i64` |
| `f16` | `f16` |
| `f32` | `f32` |
| `f64` | `f64` |
| `index` | `index` |

`ASTValue` pairs an `ir.Value` with its `ASTType` so the semantic layer can
reason about types at every stage.

### 3. Semantic analysis

The `SemanticAnalyzer` protocol controls:

- **`infer_binop_type`** -- decides the result type of a binary op (widening,
  float promotion).
- **`coerce`** -- inserts cast ops (`arith.extsi`, `arith.index_cast`,
  `emitc.cast`, etc.) when types don't match.
- **`check_return`** -- validates return types against the function
  signature.
- **`materialize_constant`** -- turns a Python `int`/`float` into an IR
  constant (`arith.constant` or `emitc.constant`).

`DefaultSemantic` implements these for the `arith` dialect.
`EmitCSemantic` implements them for the `emitc` dialect.

### 4. Scoping model

Two-tier name resolution mirrors Python semantics while supporting IR-level
isolation:

- **`SymbolTable`** -- layered table for global/function-level symbols
  (functions, classes, types).  Supports `fork()` for child tables that
  shadow the parent.
- **`ScopeStack`** -- manages a stack of `Scope` objects for local variable
  bindings.  `push_scope()` creates a nested scope; `push_scope(isolated=
  True)` hides parent locals (used for emission context blocks).
- Names are resolved locals-first, then symbols.

### 5. Escape and reflect

Two mechanisms for interleaving compile-time Python execution with IR
emission:

```python
from aster.emitter import escape, reflect

@escape
def log_value(ectx, val):
    print(f"Compile-time value: {val}")

@jit()
def f(a: i32) -> i32:
    log_value(a)                    # runs at compile time
    reflect(some_python_fn, a, 42)  # calls some_python_fn(ectx, a, 42)
    return a
```

- **`@escape`** -- marks a Python function to be called at compile time when
  invoked inside `@jit` code.  The function optionally receives the
  `EmitterContext` (detected via type annotation).
- **`reflect`** -- calls an arbitrary Python function at compile time, passing
  it the emitter context and resolved arguments.

### 6. Templates and AST rewrites

```python
from aster.emitter import template, instantiate, Type

@template(T=Type)
@jit()
def generic_add(a: T, b: T) -> T:
    return a + b

concrete = instantiate(generic_add, T=i32)
module = concrete.emit([i32, i32])
```

- **`@template`** -- declares template parameters (types, values, or
  callables).
- **`instantiate`** -- substitutes concrete values into the template AST via
  `TemplateSubstituter` and returns a new `JITFunction`.  Results are cached
  by substitution signature.
- **AST rewrites** -- `register_rewrite(rewrite_fn)` registers a global AST
  rewrite; per-jit rewrites can also be attached.  The `RewritePipeline` runs
  all rewrites before emission.

### 7. Emission contexts

Nested decorators switch the emitter table and semantic analyser for a
lexical scope:

```python
@jit()
def f(a: i32, b: i32) -> i32:
    @emitc()
    def inner():
        c = a + b   # emits emitc.add instead of arith.addi

    return a + b     # still arith.addi
```

Contexts are registered with `register_context("name", ContextClass)` and
activated with `@aster.<name>()` inside `@jit` code.  Built-in contexts:

#### 7a. EmitC context (`@emitc`)

Produces `emitc.*` ops for C++ code generation.  Covers:

- All binary, unary, and comparison ops (emitc.add, emitc.sub, etc.).
- `emitc.func` and `emitc.return` for functions.
- `emitc.if` / `emitc.for` for control flow.
- `emitc.constant` for constant materialisation.
- `emitc.cast` for type coercion.
- **Class emission**: Python `class Foo:` with annotated fields and methods
  maps to `emitc.class` with `emitc.field` and nested `emitc.func` ops.
  `self` is stripped, and `self.x` rewrites to `emitc.get_field @x`.
- **Function calls**: unresolved names produce `emitc.call_opaque` ops.
- **Method calls**: `obj.method(args)` on `_MethodReceiver` sentinels
  produces `pattern.method_call` ops.

#### 7b. Pattern context (`@pattern`)

Emits `pattern.*` ops for defining C++ rewrite patterns that lower to
`OpRewrite<T>` structs.

```python
from aster.emitter import pattern

@pattern(benefit=1, op="MyOp", fields=["counter: i32"])
def my_pat(op, rewriter):
    v = counter               # pattern.get_field @counter
    cond = v == 0
    rewriter.eraseOp(op)       # pattern.method_call @eraseOp

    def rewrite(cond):         # pattern.action
        doSomething()          # emitc.call_opaque "doSomething"
```

- `@pattern(benefit, op, fields)` captures the function and emits
  `pattern.rewrite_pattern` with optional `fields` region and required
  `body` region.
- Field declarations (`"name: type"` strings) become `pattern.field` ops.
  Bare field names in the body resolve to `pattern.get_field`.
- `def rewrite(cond):` inside the body emits `pattern.action` (guarded by
  the condition).
- `op` and `rewriter` parameters are bound as `_MethodReceiver` sentinels so
  that method calls on them emit `pattern.method_call`.
- Expressions inside the body use EmitC ops for C++ translation.
- Unresolved function names become `emitc.call_opaque` calls.
- Can also be used as a nested context (`@pattern()`) inside `@jit`.

### 8. Emitter table

`EmitterTable` is the central dispatch registry.  It supports parent-chain
inheritance: a child table overrides specific handlers while falling through
to the parent for everything else.

Registered handler types:

| Handler | Signature | Dispatch key |
|---|---|---|
| `BinOpEmitter` | `(ectx, lhs, rhs) -> ASTValue` | `ast.operator` subclass |
| `UnaryOpEmitter` | `(ectx, operand) -> ASTValue` | `ast.unaryop` subclass |
| `CmpOpEmitter` | `(ectx, lhs, rhs) -> ASTValue` | `ast.cmpop` subclass |
| `CallEmitter` | `(ectx, callee, args) -> ASTValue` | callee identity or type |
| `FunctionEmitter` | `(ectx, node, arg_types, emitter) -> None` | singleton |
| `ReturnEmitter` | `(ectx, values) -> None` | singleton |
| `IfEmitter` | `(ectx, cond, then, else, emitter) -> None` | singleton |
| `ForEmitter` | `(ectx, target, lb, ub, step, body, emitter) -> None` | singleton |
| `ClassEmitter` | `(ectx, node, emitter) -> None` | singleton |
| `AttributeEmitter` | `(ectx, obj, attr_name) -> Any` | receiver type |
| `NameEmitter` | `(ectx, obj) -> Any` | resolved value type |
| `MethodCallEmitter` | `(ectx, receiver, method, args) -> ASTValue?` | receiver type |
| `UnresolvedNameHandler` | `(ectx, name) -> Any` | singleton fallback |

## Pattern dialect (C++)

The Pattern dialect defines MLIR operations for describing C++ rewrite
patterns.  All ops implement EmitC interfaces (`DeclOpInterface`,
`StmtOpInterface`, or `ExprOpInterface`) so the existing MLIR-to-C++
translator can emit them directly.

### Ops

| Op | Interface | Description |
|---|---|---|
| `pattern.rewrite_pattern` | `DeclOpInterface` | Root op: symbol + benefit + target op name.  Optional `fields` region, required `body` region.  Emits `struct Name : OpRewrite<Op> { ... }`. |
| `pattern.field` | -- | Declares a struct field in the `fields` region. |
| `pattern.yield` | -- | Terminator for `fields` and `action` regions. |
| `pattern.action` | `StmtOpInterface` | Guarded action block taking an `i1` condition.  Emits `if (!cond) return failure();` followed by the body. |
| `pattern.get_field` | `ExprOpInterface` | Reads a struct field by symbol name.  Emits the field identifier. |
| `pattern.method_call` | `ExprOpInterface` | Calls a method on an object.  Infers `.` vs `->` from the receiver's MLIR type (`emitc.ptr` -> `->`, otherwise `.`). |

## File layout

```
python/aster/emitter/
  __init__.py          Public API
  core.py              ASTEmitter, JITFunction, @jit, default_table
  context.py           EmitterContext, EmissionContext, context registry
  table.py             EmitterTable with parent-chain dispatch
  scope.py             SymbolTable, Scope, ScopeStack
  semantic.py          SemanticAnalyzer protocol, DefaultSemantic
  types.py             ASTType, ScalarType, IndexType, ASTValue, aliases
  func_emitter.py      func.func / func.return handlers
  arith_emitter.py     arith dialect handlers
  scf_emitter.py       scf.if / scf.for handlers
  emitc_emitter.py     EmitC context, class emission, call/method sentinels
  pattern_emitter.py   Pattern context, @pattern decorator
  escape.py            @escape, reflect
  template.py          @template, instantiate, TemplateSubstituter
  rewrite.py           ASTRewrite protocol, RewritePipeline

python/aster/dialects/
  pattern.py           Python binding wrapper for pattern dialect
  PatternOps.td        TableGen wrapper for Python binding generation

include/aster/Dialect/Pattern/IR/
  PatternDialect.td    Dialect definition
  PatternOps.td        Op definitions (yield, rewrite_pattern, field,
                       action, get_field, method_call)

lib/Dialect/Pattern/IR/
  PatternDialect.cpp   Dialect registration
  PatternOps.cpp       Verifiers + EmitC interface implementations

test/Dialect/Pattern/IR/
  roundtrip.mlir       Parser/printer round-trip tests
  translate-to-cpp.mlir  C++ translation FileCheck tests
  invalid.mlir         Diagnostic verification tests

python/test/
  test_emitter.py      61 tests covering all features
```

## Test suite

61 tests in `python/test/test_emitter.py` covering:

- Scope and symbol table mechanics (7 tests)
- Emitter table parent-chain dispatch (2 tests)
- Function emission and arithmetic (7 tests)
- Escape and reflect (2 tests)
- Emission context nesting and inheritance (3 tests)
- Semantic coercion (1 test)
- Templates and instantiation (4 tests)
- AST rewrites (1 test)
- Custom emitter tables (1 test)
- Multi-op chains and augmented assignment (2 tests)
- `scf.if` control flow (3 tests)
- `scf.for` loops (7 tests)
- EmitC context ops (5 tests)
- EmitC class emission (4 tests)
- Pattern dialect emission (5 tests)
- Function and method calls (5 tests)

3 MLIR lit tests for the Pattern dialect (roundtrip, translate-to-cpp,
invalid diagnostics).
