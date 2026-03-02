//===- AsterTestDialect.cpp - test dialect --------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsterTestDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "AsterTestDialect.cpp.inc"

// Test normal form attribute.
#define GET_ATTRDEF_CLASSES
#include "TestNormalFormAttr.cpp.inc"

void mlir::aster::test::AsterTestDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestNormalFormAttr.cpp.inc"
      >();
};

namespace mlir::aster::test {
void registerAsterTestDialect(DialectRegistry &registry) {
  registry.insert<AsterTestDialect>();
}
} // namespace mlir::aster::test

using namespace mlir::aster::test;

//-----------------------------------------------------------------------------
// NoIndexTypesAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult
NoIndexTypesAttr::verifyType(llvm::function_ref<InFlightDiagnostic()> emitError,
                             Type type) const {
  if (!type)
    return llvm::success();

  if (type.isIndex())
    return emitError() << "normal form prohibits index types";

  return llvm::success();
}

//-----------------------------------------------------------------------------
// NoInvalidOpsAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult NoInvalidOpsAttr::verifyOperation(
    llvm::function_ref<InFlightDiagnostic()> emitError, Operation *op) const {
  if (isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(op))
    return emitError() << "normal form prohibits division operations";

  return llvm::success();
}

//-----------------------------------------------------------------------------
// NoInvalidAttrsAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult NoInvalidAttrsAttr::verifyAttribute(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute attr) const {
  if (!attr)
    return llvm::success();

  if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
    if (strAttr.getValue() == "invalid")
      return emitError()
             << "normal form prohibits 'invalid' string attribute values";
  }

  return llvm::success();
}
