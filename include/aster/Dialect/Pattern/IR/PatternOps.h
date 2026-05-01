//===- PatternOps.h ------------------------------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PATTERN_IR_PATTERNOPS_H
#define ASTER_DIALECT_PATTERN_IR_PATTERNOPS_H

#include "aster/Dialect/Pattern/IR/PatternDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitCInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/Pattern/IR/PatternOps.h.inc"

#endif // ASTER_DIALECT_PATTERN_IR_PATTERNOPS_H
