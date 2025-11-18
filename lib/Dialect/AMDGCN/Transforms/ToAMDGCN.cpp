//===- ToAMDGCN.cpp -------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aster {
namespace amdgcn {
#define GEN_PASS_DEF_TOAMDGCN
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//

struct ToAMDGCN : public amdgcn::impl::ToAMDGCNBase<ToAMDGCN> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Helper to create an allocation for a given register type.
static Value createAllocation(RewriterBase &rewriter, Location loc,
                              Type regTy) {
  auto rTy = cast<RegisterTypeInterface>(regTy);
  assert((isa<AGPRType, AGPRRangeType, SGPRType, SGPRRangeType, VGPRType,
              VGPRRangeType>(rTy)) &&
         "Expected a register type for allocation");
  if (!rTy.isRegisterRange())
    return amdgcn::AllocaOp::create(rewriter, loc, rTy);
  SmallVector<Value> results;
  RegisterRange range = rTy.getAsRange();
  results.reserve(range.size());
  for (int16_t i = 0; i < range.size(); ++i) {
    Value alloc = amdgcn::AllocaOp::create(
        rewriter, loc, rTy.cloneRegisterType(range.begin().getWithOffset(i)));
    results.push_back(alloc);
  }
  return MakeRegisterRangeOp::create(rewriter, loc, rTy, results);
}

/// Helper to get or split a value range.
static ValueRange getOrSplitRange(RewriterBase &rewriter, Location loc,
                                  ValueRange values) {
  if (values.size() != 1)
    return values;
  if (auto rTy = dyn_cast<RegisterTypeInterface>(values[0].getType());
      rTy && !rTy.isRegisterRange())
    return values;
  return SplitRegisterRangeOp::create(rewriter, loc, values[0]).getResults();
}

/// Helper to get the register type of given kind and size.
static Type getRegisterType(MLIRContext *ctx, RegisterKind kind, int16_t size) {
  switch (kind) {
  case amdgcn::RegisterKind::AGPR:
    return size == 1 ? Type(amdgcn::AGPRType::get(ctx, Register()))
                     : Type(amdgcn::AGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  case amdgcn::RegisterKind::SGPR:
    return size == 1 ? Type(amdgcn::SGPRType::get(ctx, Register()))
                     : Type(amdgcn::SGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  case amdgcn::RegisterKind::VGPR:
    return size == 1 ? Type(amdgcn::VGPRType::get(ctx, Register()))
                     : Type(amdgcn::VGPRRangeType::get(
                           ctx, RegisterRange(Register(), size)));
  }
}

//===----------------------------------------------------------------------===//
// PDLL rewriter helpers
//===----------------------------------------------------------------------===//

/// Helper to get the type of a value.
static LogicalResult getType(PatternRewriter &, PDLResultList &results,
                             llvm::ArrayRef<PDLValue> ins) {
  results.push_back(ins.front().cast<Value>().getType());
  return success();
}

/// Helper to get the type of a value.
static LogicalResult getValue(PatternRewriter &, PDLResultList &results,
                              llvm::ArrayRef<PDLValue> ins) {
  ValueRange values = ins[0].cast<ValueRange>();
  int64_t index = cast<IntegerAttr>(ins[1].cast<Attribute>()).getInt();
  assert(index >= 0 && index < static_cast<int64_t>(values.size()) &&
         "Index out of bounds");
  results.push_back(values[index]);
  return success();
}

/// Helper to get the opcode.
static LogicalResult getOpCode(PatternRewriter &, PDLResultList &results,
                               llvm::ArrayRef<PDLValue> ins) {
  auto attr = ins.front().cast<Attribute>();
  results.push_back(amdgcn::InstAttr::get(
      attr.getContext(),
      *amdgcn::symbolizeOpCode(cast<StringAttr>(attr).getValue())));
  return success();
}

/// Helper to get the register type of given kind and size.
static LogicalResult getReg(PatternRewriter &r, PDLResultList &results,
                            llvm::ArrayRef<PDLValue> ins) {
  std::optional<RegisterKind> kind = amdgcn::symbolizeRegisterKind(
      cast<StringAttr>(ins[0].cast<Attribute>()).getValue());
  assert(kind && "Invalid register kind");
  int64_t size = cast<IntegerAttr>(ins[1].cast<Attribute>()).getInt();
  assert(size > 0 && "Size must be positive");
  results.push_back(
      getRegisterType(r.getContext(), *kind, static_cast<int16_t>(size)));
  return success();
}

/// Helper to create an allocation for a given register type.
static LogicalResult makeAlloca(PatternRewriter &rewriter,
                                PDLResultList &results,
                                llvm::ArrayRef<PDLValue> ins) {
  Value locV = ins[0].cast<Value>();
  Type regTy = ins[1].cast<Type>();
  results.push_back(createAllocation(rewriter, locV.getLoc(), regTy));
  return success();
}

/// Helper to split a value range.
static LogicalResult makeRange(PatternRewriter &rewriter,
                               PDLResultList &results,
                               llvm::ArrayRef<PDLValue> ins) {
  auto values = ins[0].cast<ValueRange>();
  Value range =
      amdgcn::MakeRegisterRangeOp::create(rewriter, values[0].getLoc(), values);
  results.push_back(range);
  return success();
}
/// Helper to split a value range.
static LogicalResult splitRange(PatternRewriter &rewriter,
                                PDLResultList &results,
                                llvm::ArrayRef<PDLValue> ins) {
  auto values = ins[0].cast<ValueRange>();
  results.push_back(getOrSplitRange(rewriter, values[0].getLoc(), values));
  return success();
}

/// Helper to split a value range.
static LogicalResult makeConstant(PatternRewriter &rewriter,
                                  PDLResultList &results,
                                  llvm::ArrayRef<PDLValue> ins) {
  auto loc = ins[0].cast<Value>().getLoc();
  results.push_back(
      arith::ConstantOp::create(rewriter, loc,
                                cast<TypedAttr>(ins[1].cast<Attribute>()))
          .getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// PDLL constraint helpers
//===----------------------------------------------------------------------===//

/// Helper to check if a value is AGPR.
static LogicalResult isAGPR(PatternRewriter &, Value value) {
  return success(isa<amdgcn::AGPRType>(value.getType()));
}
/// Helper to check if a value is SGPR.
static LogicalResult isSGPR(PatternRewriter &, Value value) {
  return success(isa<amdgcn::SGPRType>(value.getType()));
}
/// Helper to check if a value is VGPR.
static LogicalResult isVGPR(PatternRewriter &, Value value) {
  return success(isa<amdgcn::VGPRType>(value.getType()));
}
/// Helper to check if a value is GPR.
static LogicalResult isGPR(PatternRewriter &, Value value) {
  return success(isa<amdgcn::VGPRType, amdgcn::SGPRType, amdgcn::AGPRType>(
      value.getType()));
}
/// Helper to check if a value is AGPR range of given size.
static LogicalResult isAGPRRange(PatternRewriter &, Value value,
                                 Attribute sizeAttr) {
  auto size = cast<IntegerAttr>(sizeAttr);
  if (auto rangeTy = dyn_cast<amdgcn::AGPRRangeType>(value.getType()))
    return success(rangeTy.getRange().size() == size.getInt());
  return failure();
}
/// Helper to check if a value is SGPR range of given size.
static LogicalResult isSGPRRange(PatternRewriter &, Value value,
                                 Attribute sizeAttr) {
  auto size = cast<IntegerAttr>(sizeAttr);
  if (auto rangeTy = dyn_cast<amdgcn::SGPRRangeType>(value.getType()))
    return success(rangeTy.getRange().size() == size.getInt());
  return failure();
}
/// Helper to check if a value is VGPR range of given size.
static LogicalResult isVGPRRange(PatternRewriter &, Value value,
                                 Attribute sizeAttr) {
  auto size = cast<IntegerAttr>(sizeAttr);
  if (auto rangeTy = dyn_cast<amdgcn::VGPRRangeType>(value.getType()))
    return success(rangeTy.getRange().size() == size.getInt());
  return failure();
}

/// Helper to check if a value is GPR range of given size.
static LogicalResult isGPRRange(PatternRewriter &r, Value value,
                                Attribute sizeAttr) {
  if (!isa<amdgcn::AGPRRangeType, amdgcn::SGPRRangeType, amdgcn::VGPRRangeType>(
          value.getType()))
    return failure();
  auto size = cast<IntegerAttr>(sizeAttr);
  if (auto rangeTy = dyn_cast<RegisterTypeInterface>(value.getType()))
    return success(rangeTy.getAsRange().size() == size.getInt());
  return failure();
}

//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//

#define ID _pdlModStr
#define SIZE_ID _pdlModStrSize
#include "aster/Dialect/AMDGCN/Transforms/InstSelection.mlir.inc"
#undef SIZE_ID
#undef ID

static OwningOpRef<mlir::ModuleOp> getPDLModule(StringRef inputFile,
                                                MLIRContext *ctx) {
  if (!inputFile.empty()) {
    return parseSourceFile<mlir::ModuleOp>(inputFile, ParserConfig(ctx));
  }
  StringRef data(_pdlModStr, _pdlModStrSize);
  return parseSourceString<mlir::ModuleOp>(data, ParserConfig(ctx));
}

void ToAMDGCN::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  OwningOpRef<mlir::ModuleOp> moduleRef =
      getPDLModule(inputFile, &getContext());
  if (!moduleRef) {
    op->emitError("failed to parse input module from file: ") << inputFile;
    return signalPassFailure();
  }
  PDLPatternModule pdlPattern(std::move(moduleRef));
  pdlPattern.registerRewriteFunction("get_type", getType);
  pdlPattern.registerRewriteFunction("get_value", getValue);
  pdlPattern.registerRewriteFunction("get_opcode", getOpCode);
  pdlPattern.registerRewriteFunction("get_reg", getReg);
  pdlPattern.registerRewriteFunction("make_alloca", makeAlloca);
  pdlPattern.registerRewriteFunction("make_range", makeRange);
  pdlPattern.registerRewriteFunction("split_range", splitRange);
  pdlPattern.registerRewriteFunction("constant", makeConstant);
  pdlPattern.registerConstraintFunction("IsAGPR", isAGPR);
  pdlPattern.registerConstraintFunction("IsSGPR", isSGPR);
  pdlPattern.registerConstraintFunction("IsVGPR", isVGPR);
  pdlPattern.registerConstraintFunction("IsGPR", isGPR);
  pdlPattern.registerConstraintFunction("IsAGPRRange", isAGPRRange);
  pdlPattern.registerConstraintFunction("IsSGPRRange", isSGPRRange);
  pdlPattern.registerConstraintFunction("IsVGPRRange", isVGPRRange);
  pdlPattern.registerConstraintFunction("IsGPRRange", isGPRRange);
  patterns.add(std::move(pdlPattern));
  if (failed(applyPatternsGreedily(
          op, FrozenRewritePatternSet(std::move(patterns)))))
    return signalPassFailure();
}
