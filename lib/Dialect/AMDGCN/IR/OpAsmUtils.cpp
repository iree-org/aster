//===- OpAsmUtils.cpp - AMDGCN assembly parse/print helpers ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/OpAsmUtils.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

ParseResult amdgcn::parseDimAttr(OpAsmParser &parser, DimAttr &attr) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();

  auto dimOpt = symbolizeDim(keyword);
  if (!dimOpt)
    return parser.emitError(parser.getCurrentLocation(), "invalid dimension: ")
           << keyword;

  attr = DimAttr::get(parser.getBuilder().getContext(), *dimOpt);
  return success();
}

void amdgcn::printDimAttr(OpAsmPrinter &printer, Operation *, DimAttr attr) {
  printer << stringifyDim(attr.getValue());
}

ParseResult amdgcn::parseLoadResults(OpAsmParser &parser, Type destType,
                                     Type &destResType, Type &tokenType) {
  auto regTy = dyn_cast<RegisterTypeInterface>(destType);
  if (regTy && regTy.hasValueSemantics())
    destResType = destType;
  if (parser.parseType(tokenType))
    return failure();
  return success();
}

void amdgcn::printLoadResults(OpAsmPrinter &printer, Operation *, Type destType,
                              Type destResType, Type tokenType) {
  (void)destType;
  (void)destResType;
  printer.printType(tokenType);
}

ParseResult amdgcn::parseAllocSize(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &dynamicSize,
    IntegerAttr &staticSize) {
  int64_t intVal;
  auto intRes = parser.parseOptionalInteger(intVal);
  if (intRes.has_value()) {
    if (failed(*intRes))
      return failure();
    staticSize = parser.getBuilder().getI64IntegerAttr(intVal);
    dynamicSize = std::nullopt;
    return success();
  }

  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();
  dynamicSize = operand;
  staticSize = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
  return success();
}

void amdgcn::printAllocSize(OpAsmPrinter &printer, Operation *op,
                            Value dynamicSize, IntegerAttr staticSize) {
  (void)op;
  if (ShapedType::isDynamic(staticSize.getInt())) {
    printer.printOperand(dynamicSize);
    return;
  }
  printer << staticSize.getInt();
}
