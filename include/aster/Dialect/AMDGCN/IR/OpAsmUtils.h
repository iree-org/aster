//===- OpAsmUtils.h - AMDGCN assembly parse/print helpers -----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared custom assembly format helpers for TableGen-generated AMDGCN ops.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_OP_ASM_UTILS_H
#define ASTER_DIALECT_AMDGCN_IR_OP_ASM_UTILS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::aster::amdgcn {

ParseResult parseDimAttr(OpAsmParser &parser, DimAttr &attr);
void printDimAttr(OpAsmPrinter &printer, Operation *, DimAttr attr);

ParseResult parseLoadResults(OpAsmParser &parser, Type destType,
                             Type &destResType, Type &tokenType);
void printLoadResults(OpAsmPrinter &printer, Operation *, Type destType,
                      Type destResType, Type tokenType);

ParseResult
parseAllocSize(OpAsmParser &parser,
               std::optional<OpAsmParser::UnresolvedOperand> &dynamicSize,
               IntegerAttr &staticSize);
void printAllocSize(OpAsmPrinter &printer, Operation *op, Value dynamicSize,
                    IntegerAttr staticSize);

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_OP_ASM_UTILS_H
