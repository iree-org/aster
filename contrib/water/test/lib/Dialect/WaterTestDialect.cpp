// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "WaterTestDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Visitors.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::transform;

#include "WaterTestDialect.cpp.inc"

// Test normal form attribute.
#define GET_ATTRDEF_CLASSES
#include "TestWaterNormalFormAttr.cpp.inc"

void mlir::water::test::WaterTestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "WaterTestDialectOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestWaterNormalFormAttr.cpp.inc"
      >();
};

namespace mlir::water::test {
void registerWaterTestDialect(DialectRegistry &registry) {
  registry.insert<WaterTestDialect>();
}
} // namespace mlir::water::test

using namespace mlir::water::test;

//-----------------------------------------------------------------------------
// Walk helpers for checkOperation implementations.
//-----------------------------------------------------------------------------

// Walk all types of operands, results and block arguments of the given op and
// its nested operations, calling `check` on each unique type. Stops as soon as
// `check` returns failure.
static llvm::LogicalResult
walkAllTypes(Operation *root,
             llvm::function_ref<llvm::LogicalResult(Type, Location)> check) {
  llvm::SmallPtrSet<Type, 16> seenTypes;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto checkType = [&](Type type, Location loc) {
      if (!type)
        return WalkResult::advance();
      if (!seenTypes.insert(type).second)
        return WalkResult::advance();
      if (llvm::failed(check(type, loc)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    };
    for (Type type : op->getOperandTypes())
      if (checkType(type, op->getLoc()).wasInterrupted())
        return WalkResult::interrupt();
    for (Type type : op->getResultTypes())
      if (checkType(type, op->getLoc()).wasInterrupted())
        return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (checkType(arg.getType(), arg.getLoc()).wasInterrupted())
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return llvm::failure(walkResult.wasInterrupted());
}

// Walk all attributes attached to the given op and any nested operations,
// recursing into composite attributes/types using AttrTypeWalker. Stops as
// soon as `check` returns failure.
static llvm::LogicalResult walkAllAttributes(
    Operation *root,
    llvm::function_ref<llvm::LogicalResult(Attribute, Location)> check) {
  llvm::SmallPtrSet<Attribute, 16> seenAttrs;
  AttrTypeWalker walker;
  Location loc = root->getLoc();
  walker.addWalk([&](Attribute attr) {
    if (!seenAttrs.insert(attr).second)
      return WalkResult::skip();
    if (llvm::failed(check(attr, loc)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  WalkResult walkResult = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    loc = op->getLoc();
    for (NamedAttribute attr : op->getAttrs()) {
      if (walker.walk(attr.getValue()).wasInterrupted())
        return WalkResult::interrupt();
    }
    for (OpResult result : op->getResults()) {
      if (walker.walk(result.getType()).wasInterrupted())
        return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (walker.walk(arg.getType()).wasInterrupted())
            return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return llvm::failure(walkResult.wasInterrupted());
}

//-----------------------------------------------------------------------------
// NoIndexTypesAttr interface implementations.
//-----------------------------------------------------------------------------

DiagnosedSilenceableFailure
NoIndexTypesAttr::checkOperation(Operation *op) const {
  if (llvm::failed(walkAllTypes(op, [](Type type, Location loc) {
        if (type.isIndex()) {
          emitError(loc) << "normal form prohibits index types";
          return llvm::failure();
        }
        return llvm::success();
      })))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

//-----------------------------------------------------------------------------
// NoInvalidOpsAttr interface implementations.
//-----------------------------------------------------------------------------

DiagnosedSilenceableFailure
NoInvalidOpsAttr::checkOperation(Operation *op) const {
  WalkResult walkResult = op->walk([&](Operation *nested) {
    if (isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(nested)) {
      nested->emitError() << "normal form prohibits division operations";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

//-----------------------------------------------------------------------------
// NoInvalidAttrsAttr interface implementations.
//-----------------------------------------------------------------------------

DiagnosedSilenceableFailure
NoInvalidAttrsAttr::checkOperation(Operation *op) const {
  if (llvm::failed(walkAllAttributes(op, [](Attribute attr, Location loc) {
        if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
          if (strAttr.getValue() == "invalid") {
            emitError(loc)
                << "normal form prohibits 'invalid' string attribute values";
            return llvm::failure();
          }
        }
        return llvm::success();
      })))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

//-----------------------------------------------------------------------------
// NoForbiddenSymbolsAttr interface implementations.
//-----------------------------------------------------------------------------

DiagnosedSilenceableFailure
NoForbiddenSymbolsAttr::checkOperation(Operation *op) const {
  if (llvm::failed(
          walkAllAttributes(op, [](Attribute attr, Location loc) {
            if (auto symbolAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(attr)) {
              if (symbolAttr.getName() == "forbidden") {
                emitError(loc)
                    << "normal form prohibits 'forbidden' symbol in types";
                return llvm::failure();
              }
            }
            return llvm::success();
          })))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

//-----------------------------------------------------------------------------
// WaveFailPropagationOp implementations.
//-----------------------------------------------------------------------------

llvm::FailureOr<ChangeResult> WaveFailPropagationOp::propagateForward(
    llvm::ArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getForward()) {
    errs << "intentionally failed to propagate forward";
    return failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      operandTypes, resultTypes, "operands", "results", errs);
}

llvm::FailureOr<ChangeResult> WaveFailPropagationOp::propagateBackward(
    llvm::MutableArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getBackward()) {
    errs << "intentionally failed to propagate backward";
    return failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      resultTypes, operandTypes, "results", "operands", errs);
}

LogicalResult WaveFailPropagationOp::finalizeTypeInference() {
  return success();
}

#define GET_OP_CLASSES
#include "WaterTestDialectOps.cpp.inc"
