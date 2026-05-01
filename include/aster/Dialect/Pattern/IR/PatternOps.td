//===- PatternOps.td - Pattern operations -----------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PATTERN_IR_PATTERNOPS_TD
#define ASTER_DIALECT_PATTERN_IR_PATTERNOPS_TD

include "aster/Dialect/Pattern/IR/PatternDialect.td"
include "mlir/Dialect/EmitC/IR/EmitCInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Pattern_YieldOp : Pattern_Op<"yield", [Terminator]> {
  let summary = "Yield terminator for pattern regions";
  let description = [{
    Terminates a `fields` or `action` region. In a `fields` region the
    operands correspond to the declared fields. In an `action` region the
    op takes no operands.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [
    OpBuilder<(ins), [{ /* no-op: default empty yield */ }]>
  ];
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

//===----------------------------------------------------------------------===//
// RewritePatternOp
//===----------------------------------------------------------------------===//

def Pattern_RewritePatternOp : Pattern_Op<"rewrite_pattern", [
    DeclareOpInterfaceMethods<DeclOpInterface>,
    IsolatedFromAbove, Symbol
  ]> {
  let summary = "Define a C++ rewrite pattern struct";
  let description = [{
    Defines a rewrite pattern that translates to a C++ struct inheriting
    from `OpRewrite<OpName>`. The optional `fields` region declares struct
    member variables and the required `body` region contains the
    match-and-rewrite logic.

    Example:

    ```mlir
    pattern.rewrite_pattern @my_pat benefit(1) op(@SomeOp) fields {
      %f = pattern.field @counter : i32
      pattern.yield %f : i32
    } body {
      pattern.action %cond {
        pattern.yield
      }
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$benefit,
    FlatSymbolRefAttr:$op_name
  );
  let regions = (region MaxSizedRegion<1>:$fieldsRegion,
                        SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name `benefit` `(` $benefit `)` `op` `(` $op_name `)`
    (`fields` $fieldsRegion^)? `body` $bodyRegion attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FieldOp
//===----------------------------------------------------------------------===//

def Pattern_FieldOp : Pattern_Op<"field", [Pure]> {
  let summary = "Declare a struct field in a rewrite pattern";
  let description = [{
    Declares a field that becomes a member variable of the generated C++
    struct. The symbol name becomes the C++ identifier and the result type
    determines the C++ type.

    Example:

    ```mlir
    %f = pattern.field @my_field : i32
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ActionOp
//===----------------------------------------------------------------------===//

def Pattern_ActionOp : Pattern_Op<"action", [
    DeclareOpInterfaceMethods<StmtOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">,
    Terminator
  ]> {
  let summary = "Conditional action block in a rewrite pattern body";
  let description = [{
    Defines a guarded action. The condition must be `i1`. When translated
    to C++ the op emits:

    ```cpp
    if (!cond)
      return failure();
    // body
    ```
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$condition attr-dict-with-keyword $bodyRegion";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetFieldOp
//===----------------------------------------------------------------------===//

def Pattern_GetFieldOp : Pattern_Op<"get_field", [
    DeclareOpInterfaceMethods<ExprOpInterface>,
    Pure
  ]> {
  let summary = "Read a struct field by name";
  let description = [{
    Returns a const-qualified reference to a field declared in the
    enclosing `rewrite_pattern`'s `fields` region. The `field_name`
    must match a `pattern.field` symbol in that region.

    Example:

    ```mlir
    %v = pattern.get_field @counter : i32
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$field_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$field_name attr-dict `:` type($result)";
}

#endif // ASTER_DIALECT_PATTERN_IR_PATTERNOPS_TD
