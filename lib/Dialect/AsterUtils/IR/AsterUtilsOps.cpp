//===- AsterUtilsOps.cpp - AsterUtils operations ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

//===----------------------------------------------------------------------===//
// AsterUtils Inliner Interface
//===----------------------------------------------------------------------===//

namespace {
struct AsterUtilsInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Always allow inlining of AsterUtils operations.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Always allow inlining of AsterUtils operations into regions.
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
// AsterUtils dialect
//===----------------------------------------------------------------------===//

void AsterUtilsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
  addInterfaces<AsterUtilsInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// AssumeRangeOp
//===----------------------------------------------------------------------===//

OpFoldResult AssumeRangeOp::fold(FoldAdaptor adaptor) {
  if (!getMin().has_value() && !getMax().has_value())
    return getInput();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

void ExecuteRegionOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }
  regions.push_back(
      RegionSuccessor(getOperation()->getParentOp(), getResults()));
}

//===----------------------------------------------------------------------===//
// FromAnyOp
//===----------------------------------------------------------------------===//

/// Fold FromAnyOp(ToAnyOp(x)) to x when the types match.
OpFoldResult FromAnyOp::fold(FoldAdaptor adaptor) {
  Value value;
  auto toAny = getInput().getDefiningOp<ToAnyOp>();
  while (toAny) {
    if (toAny.getInput().getType() != getType())
      break;
    value = toAny.getInput();
    auto fromAny = value.getDefiningOp<FromAnyOp>();
    if (!fromAny)
      break;
    toAny = fromAny.getInput().getDefiningOp<ToAnyOp>();
  }
  return value;
}

//===----------------------------------------------------------------------===//
// ToAnyOp
//===----------------------------------------------------------------------===//

/// Fold ToAnyOp(FromAnyOp(x)) to x when the types match.
OpFoldResult ToAnyOp::fold(FoldAdaptor adaptor) {
  Value value;
  Type type = getInput().getType();
  auto fromAny = getInput().getDefiningOp<FromAnyOp>();
  while (fromAny) {
    if (fromAny.getType() != type)
      break;
    auto toAny = fromAny.getInput().getDefiningOp<ToAnyOp>();
    if (!toAny || toAny.getInput().getType() != type)
      break;
    value = toAny;
    fromAny = toAny.getInput().getDefiningOp<FromAnyOp>();
  }
  return value;
}

//===----------------------------------------------------------------------===//
// IncGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.cpp.inc"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.cpp.inc"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsEnums.cpp.inc"
