//===- PatternDialect.td - Pattern dialect ----------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PATTERN_IR_PATTERNDIALECT_TD
#define ASTER_DIALECT_PATTERN_IR_PATTERNDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pattern dialect definition
//===----------------------------------------------------------------------===//

def Pattern_Dialect : Dialect {
  let name = "pattern";
  let summary = "Dialect for generating C++ rewrite pattern structs via EmitC";
  let description = [{
    The Pattern dialect provides operations for defining MLIR rewrite patterns
    that are translated to C++ structs inheriting from `OpRewrite`. Each
    pattern declares optional struct fields and a match-and-rewrite body.
    The dialect implements EmitC interfaces (DeclOpInterface,
    StmtOpInterface, ExprOpInterface) for direct C++ code emission.
  }];
  let cppNamespace = "::mlir::aster::pattern";
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class Pattern_Op<string mnemonic, list<Trait> traits = []>
    : Op<Pattern_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_PATTERN_IR_PATTERNDIALECT_TD
