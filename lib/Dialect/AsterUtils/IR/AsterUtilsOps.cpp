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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

/// Parse a bound that can be either static (integer) or dynamic (SSA value).
/// Format: `keyword <integer>` for static, `keyword %operand` for dynamic.
/// Returns success even if the keyword is not present (optional bound).
static ParseResult
parseAssumeRangeBound(OpAsmParser &parser, StringRef keyword,
                      std::optional<OpAsmParser::UnresolvedOperand> &dynamic,
                      IntegerAttr &staticVal) {
  // Check if the keyword is present.
  if (failed(parser.parseOptionalKeyword(keyword))) {
    dynamic = std::nullopt;
    staticVal = nullptr;
    return success();
  }

  // Try to parse an integer first (static bound).
  int64_t intVal;
  auto intRes = parser.parseOptionalInteger(intVal);
  if (intRes.has_value()) {
    if (failed(*intRes))
      return failure();
    staticVal = parser.getBuilder().getIndexAttr(intVal);
    dynamic = std::nullopt;
    return success();
  }

  // Otherwise, parse an operand (dynamic bound).
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();
  dynamic = operand;
  staticVal = nullptr;
  return success();
}

/// Print a bound that can be either static or dynamic.
static void printAssumeRangeBound(OpAsmPrinter &printer, Operation *,
                                  StringRef keyword, Value dynamic,
                                  IntegerAttr staticVal) {
  if (!dynamic && !staticVal)
    return;
  printer << " " << keyword << " ";
  if (dynamic) {
    printer.printOperand(dynamic);
    return;
  }
  printer << staticVal.getInt();
}

LogicalResult AssumeRangeOp::verify() {
  // Cannot have both static and dynamic min.
  if (getDynamicMin() && getStaticMin())
    return emitOpError("cannot have both static and dynamic min");
  // Cannot have both static and dynamic max.
  if (getDynamicMax() && getStaticMax())
    return emitOpError("cannot have both static and dynamic max");
  return success();
}

OpFoldResult AssumeRangeOp::fold(FoldAdaptor adaptor) {
  bool changed = false;
  if (auto attr = dyn_cast_if_present<IntegerAttr>(adaptor.getDynamicMax());
      attr && attr.getType().getIntOrFloatBitWidth() <= 64) {
    getDynamicMaxMutable().clear();
    setStaticMaxAttr(IntegerAttr::get(IndexType::get(getContext()),
                                      attr.getValue().getSExtValue()));
    changed = true;
  }
  if (auto attr = dyn_cast_if_present<IntegerAttr>(adaptor.getDynamicMin());
      attr && attr.getType().getIntOrFloatBitWidth() <= 64) {
    getDynamicMinMutable().clear();
    setStaticMinAttr(IntegerAttr::get(IndexType::get(getContext()),
                                      attr.getValue().getSExtValue()));
    changed = true;
  }
  return changed ? OpFoldResult(getResult()) : OpFoldResult();
}

namespace {
/// Canonicalize assume_range by folding constant dynamic bounds into static.
struct FoldConstantAssumeRangeBounds : public OpRewritePattern<AssumeRangeOp> {
  using OpRewritePattern<AssumeRangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssumeRangeOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    IntegerAttr staticMin = op.getStaticMinAttr();
    IntegerAttr staticMax = op.getStaticMaxAttr();
    Value dynamicMin = op.getDynamicMin();
    Value dynamicMax = op.getDynamicMax();

    // Try to fold constant dynamic min.
    if (dynamicMin) {
      if (auto constOp = dynamicMin.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          staticMin = rewriter.getIndexAttr(intAttr.getInt());
          dynamicMin = nullptr;
          changed = true;
        }
      }
    }

    // Try to fold constant dynamic max.
    if (dynamicMax) {
      if (auto constOp = dynamicMax.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          staticMax = rewriter.getIndexAttr(intAttr.getInt());
          dynamicMax = nullptr;
          changed = true;
        }
      }
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<AssumeRangeOp>(op, op.getType(), op.getInput(),
                                               dynamicMin, dynamicMax,
                                               staticMin, staticMax);
    return success();
  }
};
} // namespace

void AssumeRangeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<FoldConstantAssumeRangeBounds>(context);
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
