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
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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
  regions.push_back(RegionSuccessor::parent());
}

ValueRange ExecuteRegionOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.isParent())
    return getResults();
  return {};
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
// StructCreateOp
//===----------------------------------------------------------------------===//

LogicalResult StructCreateOp::verify() {
  auto structType = llvm::cast<StructType>(getResult().getType());
  ArrayRef<Type> fieldTypes = structType.getFieldTypes();

  // Check that the number of operands matches the number of fields.
  if (getFields().size() != fieldTypes.size()) {
    return emitOpError("expected ")
           << fieldTypes.size() << " field values, but got "
           << getFields().size();
  }

  // Check that each operand type matches the corresponding field type.
  for (size_t i = 0, e = fieldTypes.size(); i < e; ++i) {
    if (getFields()[i].getType() != fieldTypes[i]) {
      return emitOpError("field ")
             << i << " ('" << structType.getFieldName(i).getValue()
             << "') type mismatch: expected " << fieldTypes[i] << ", got "
             << getFields()[i].getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

LogicalResult StructExtractOp::verify() {
  auto structType = llvm::cast<StructType>(getInput().getType());
  ArrayAttr fieldNames = getFieldNames();
  ResultRange results = getResults();

  // Check that the number of field names matches the number of results.
  if (fieldNames.size() != results.size()) {
    return emitOpError("expected ")
           << fieldNames.size() << " results for " << fieldNames.size()
           << " field names, but got " << results.size();
  }

  // Check each field name and result type.
  for (size_t i = 0, e = fieldNames.size(); i < e; ++i) {
    auto fieldName = llvm::cast<StringAttr>(fieldNames[i]).getValue();

    // Check that the field name exists in the struct type.
    auto fieldIndex = structType.getFieldIndex(fieldName);
    if (!fieldIndex) {
      return emitOpError("field '")
             << fieldName << "' does not exist in struct type " << structType;
    }

    // Check that the result type matches the field type.
    Type expectedType = structType.getFieldType(*fieldIndex);
    if (results[i].getType() != expectedType) {
      return emitOpError("result type mismatch: field '")
             << fieldName << "' has type " << expectedType << ", but got "
             << results[i].getType();
    }
  }

  return success();
}

namespace {
/// Fold struct_extract(struct_create(...)) to the corresponding operands.
struct FoldStructExtractOfCreate : public OpRewritePattern<StructExtractOp> {
  using OpRewritePattern<StructExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StructExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto createOp = op.getInput().getDefiningOp<StructCreateOp>();
    if (!createOp)
      return failure();

    auto structType = llvm::cast<StructType>(op.getInput().getType());
    ArrayAttr fieldNames = op.getFieldNames();

    // Map each extracted field to its corresponding operand in struct_create.
    SmallVector<Value> replacements;
    for (Attribute attr : fieldNames) {
      auto fieldName = llvm::cast<StringAttr>(attr).getValue();
      auto fieldIndex = structType.getFieldIndex(fieldName);
      assert(fieldIndex && "field name should exist (verified)");
      replacements.push_back(createOp.getFields()[*fieldIndex]);
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

void StructExtractOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FoldStructExtractOfCreate>(context);
}

//===----------------------------------------------------------------------===//
// IncGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.cpp.inc"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.cpp.inc"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsEnums.cpp.inc"
