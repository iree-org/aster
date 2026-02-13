//===- LSIROps.cpp - LSIR operations ----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
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
using namespace mlir::aster::lsir;

//===----------------------------------------------------------------------===//
// LSIR Inliner Interface
//===----------------------------------------------------------------------===//

namespace {
struct LSIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Always allow inlining of LSIR operations.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Always allow inlining of LSIR operations into regions.
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
// LSIR dialect
//===----------------------------------------------------------------------===//

void LSIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/LSIR/IR/LSIROps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
  addInterfaces<LSIRInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// LSIR Operation Verifiers
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
// LSIR LoadOp
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
// LSIR StoreOp
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
// LSIR CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::canonicalize(CopyOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  TypedValue<RegisterTypeInterface> tgt = op.getTarget();
  TypedValue<RegisterTypeInterface> src = op.getSource();
  if (tgt == src || (tgt.getType().hasAllocatedSemantics() &&
                     src.getType().hasAllocatedSemantics() &&
                     tgt.getType() == src.getType())) {
    if (op.getTargetRes())
      rewriter.replaceOp(op, op.getSource());
    else
      rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// LSIR RegCastOp
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
// LSIR IncGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/LSIR/IR/LSIROps.cpp.inc"

#include "aster/Dialect/LSIR/IR/LSIRDialect.cpp.inc"

#include "aster/Dialect/LSIR/IR/LSIREnums.cpp.inc"
