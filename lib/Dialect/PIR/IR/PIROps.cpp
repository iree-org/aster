//===- PIROps.cpp - PIR operations ------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

//===----------------------------------------------------------------------===//
// PIR Inliner Interface
//===----------------------------------------------------------------------===//

namespace {
struct PIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Always allow inlining of PIR operations.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Always allow inlining of PIR operations into regions.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    return true;
  }

  /// Always allow inlining of regions.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PIR dialect
//===----------------------------------------------------------------------===//

void PIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/PIR/IR/PIROps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
  addInterfaces<PIRInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// PIR Operation Verifiers
//===----------------------------------------------------------------------===//

LogicalResult AssumeNoaliasOp::verify() {
  // No two operands are the same
  auto operands = getOperands();
  for (size_t i = 0; i < operands.size(); ++i) {
    for (size_t j = i + 1; j < operands.size(); ++j) {
      if (operands[i] == operands[j]) {
        return emitOpError("operand ")
               << i << " and operand " << j << " must be different";
      }
    }
  }

  // Inputs and outputs have the same size
  if (getOperands().size() != getResults().size()) {
    return emitOpError("number of operands (")
           << getOperands().size() << ") must match number of results ("
           << getResults().size() << ")";
  }

  // Every input has the same type as its matching output
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getOperands(), getResults()))) {
    auto [operand, result] = pair;
    if (operand.getType() != result.getType()) {
      return emitOpError("operand ")
             << idx << " type " << operand.getType() << " must match result "
             << idx << " type " << result.getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PIR AssumeRangeOp
//===----------------------------------------------------------------------===//

OpFoldResult AssumeRangeOp::fold(FoldAdaptor adaptor) {
  if (!getMin().has_value() && !getMax().has_value())
    return getInput();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PIR LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  ptr::MemorySpaceAttrInterface memorySpace = getMemorySpace();
  auto emitError = [&]() -> InFlightDiagnostic { return emitOpError(); };

  // Check if load is valid for this memory space
  if (!memorySpace.isValidLoad(getDst().getType(),
                               ptr::AtomicOrdering::not_atomic,
                               /*alignment=*/std::nullopt,
                               /*dataLayout=*/nullptr, emitError)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PIR StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  ptr::MemorySpaceAttrInterface memorySpace = getMemorySpace();
  auto emitError = [&]() -> InFlightDiagnostic { return emitOpError(); };

  // Check if store is valid for this memory space
  if (!memorySpace.isValidStore(getValue().getType(),
                                ptr::AtomicOrdering::not_atomic,
                                /*alignment=*/std::nullopt,
                                /*dataLayout=*/nullptr, emitError)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PIR RegCastOp
//===----------------------------------------------------------------------===//

OpFoldResult RegCastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType())
    return getSrc();
  auto src = dyn_cast_if_present<RegCastOp>(getSrc().getDefiningOp());
  while (src != nullptr) {
    if (getType() == src.getSrc().getType())
      return src.getSrc();
    src = dyn_cast_if_present<RegCastOp>(src.getSrc().getDefiningOp());
  }
  return nullptr;
}

LogicalResult RegCastOp::canonicalize(RegCastOp op,
                                      ::mlir::PatternRewriter &rewriter) {
  if (op.getType() == op.getSrc().getType()) {
    rewriter.replaceOp(op, op.getSrc());
    return success();
  }
  Value src = op.getSrc();
  auto cOp = dyn_cast_if_present<RegCastOp>(op.getSrc().getDefiningOp());
  while (cOp != nullptr) {
    src = cOp.getSrc();
    cOp = dyn_cast_if_present<RegCastOp>(cOp.getSrc().getDefiningOp());
  }
  if (src != op.getSrc()) {
    auto newOp = RegCastOp::create(rewriter, op.getLoc(), op.getType(), src);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// PIR IncGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/PIR/IR/PIROps.cpp.inc"

#include "aster/Dialect/PIR/IR/PIRDialect.cpp.inc"

#include "aster/Dialect/PIR/IR/PIREnums.cpp.inc"
