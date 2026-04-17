// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/X86/IR/X86Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::x86;

#include "aster/Dialect/X86/IR/Interfaces/X86AsmOpInterface.cpp.inc"
#include "aster/Dialect/X86/IR/Interfaces/X86IsaOpInterface.cpp.inc"

#include "aster/Dialect/X86/IR/X86Dialect.cpp.inc"
#include "aster/Dialect/X86/IR/X86Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Types.cpp.inc"

// Custom parser/printer for bare-keyword instruction names.
static ParseResult parseBareKeyword(OpAsmParser &parser, StringAttr &inst) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  inst = parser.getBuilder().getStringAttr(keyword);
  return success();
}

static void printBareKeyword(OpAsmPrinter &printer, Operation *,
                             StringAttr inst) {
  printer << inst.getValue();
}

#define GET_OP_CLASSES
#include "aster/Dialect/X86/IR/X86Ops.cpp.inc"

void X86Dialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/X86/IR/X86Attrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/X86/IR/X86Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/X86/IR/X86Ops.cpp.inc"
      >();
}

LogicalResult ModuleOp::verifyRegions() {
  IsaVersion target = getTargetIsa();
  LogicalResult result = success();
  getBodyRegion().walk([&](Operation *op) -> WalkResult {
    auto isaOp = dyn_cast<X86IsaOpInterface>(op);
    if (!isaOp)
      return WalkResult::advance();
    IsaVersion required = isaOp.getRequiredIsa();
    if (static_cast<int32_t>(required) > static_cast<int32_t>(target)) {
      result = isaOp->emitOpError("requires ")
               << stringifyIsaVersion(required)
               << " but enclosing x86.module targets "
               << stringifyIsaVersion(target);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}
