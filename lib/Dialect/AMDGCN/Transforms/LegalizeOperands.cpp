//===- LegalizeOperands.cpp - Legalize operand constraints ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legalizes operand constraints unsatisfiable by hardware encoding, e.g.
// SOP2 at-most-one-literal and VOP2 src1-must-be-VGPR.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LEGALIZEOPERANDS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

struct LegalizeOperands
    : public amdgcn::impl::LegalizeOperandsBase<LegalizeOperands> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// VGPRSelectInstLegalizePattern
//===----------------------------------------------------------------------===//

// Handles VALU select ops (non-SCC condition) where the true-value operand
// must be materialized into a VGPR (VOP2 src1 must be VGPR, VOP3 must be
// inline).
struct VGPRSelectInstLegalizePattern
    : public OpInterfaceRewritePattern<SelectInstOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(SelectInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SGPRSelectInstLegalizePattern
//===----------------------------------------------------------------------===//

// Handles SALU select ops (SCC condition) where two non-inline constants
// require materializing the true-value operand (SOP2 at-most-one-literal rule).
struct SGPRSelectInstLegalizePattern
    : public OpInterfaceRewritePattern<SelectInstOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(SelectInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

} // namespace

/// Return the integer value of an arith.constant with the given bit width.
static std::optional<int64_t> getConstInt(Value v, int width) {
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
      if (intAttr.getType().isSignlessInteger(static_cast<unsigned>(width)))
        return intAttr.getInt();
  return std::nullopt;
}

/// Inline integer constant range [-16, 64] for AMD GCN.
static bool isInlineInt(int64_t val) { return val >= -16 && val <= 64; }

//===----------------------------------------------------------------------===//
// VGPRSelectInstLegalizePattern
//===----------------------------------------------------------------------===//

LogicalResult VGPRSelectInstLegalizePattern::matchAndRewrite(
    SelectInstOpInterface op, PatternRewriter &rewriter) const {
  if (isa<SCCType>(op.getConditionOperand().getType()))
    return failure();
  Value trueVal = op.getTrueValueOperand().getValue();
  std::optional<int64_t> trueConst = getConstInt(trueVal, 32);
  if (!trueConst || isInlineInt(*trueConst))
    return failure();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  Value out = createAllocation(rewriter, loc, getVGPR(ctx));
  Value movResult = VMovB32::create(rewriter, loc, out, trueVal).getDst0Res();
  rewriter.modifyOpInPlace(
      op, [&]() { op.getTrueValueOperand().get()->set(movResult); });
  return success();
}

//===----------------------------------------------------------------------===//
// SGPRSelectInstLegalizePattern
//===----------------------------------------------------------------------===//

LogicalResult SGPRSelectInstLegalizePattern::matchAndRewrite(
    SelectInstOpInterface op, PatternRewriter &rewriter) const {
  if (!isa<SCCType>(op.getConditionOperand().getType()))
    return failure();
  Value trueVal = op.getTrueValueOperand().getValue();
  Value falseVal = op.getFalseValueOperand().getValue();
  // Try 32-bit first, then 64-bit.
  std::optional<int64_t> trueConst = getConstInt(trueVal, 32);
  std::optional<int64_t> falseConst = getConstInt(falseVal, 32);
  int bitWidth = 32;
  if (!trueConst && !falseConst) {
    trueConst = getConstInt(trueVal, 64);
    falseConst = getConstInt(falseVal, 64);
    bitWidth = 64;
  }
  if (!trueConst || !falseConst || isInlineInt(*trueConst) ||
      isInlineInt(*falseConst))
    return failure();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  Value movResult;
  if (bitWidth == 32) {
    RegisterTypeInterface sgprTy = getSGPR(ctx);
    Value out = createAllocation(rewriter, loc, sgprTy);
    movResult =
        SMovB32::create(rewriter, loc, sgprTy, out, trueVal).getDst0Res();
  } else {
    RegisterTypeInterface sgprTy = getSGPR(ctx, 2);
    Value out = createAllocation(rewriter, loc, sgprTy);
    movResult =
        SMovB64::create(rewriter, loc, sgprTy, out, trueVal).getDst0Res();
  }
  rewriter.modifyOpInPlace(
      op, [&]() { op.getTrueValueOperand().get()->set(movResult); });
  return success();
}

//===----------------------------------------------------------------------===//
// LegalizeOperands pass
//===----------------------------------------------------------------------===//

void LegalizeOperands::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<VGPRSelectInstLegalizePattern, SGPRSelectInstLegalizePattern>(
      &getContext());
  if (failed(applyPatternsGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
    signalPassFailure();
}
