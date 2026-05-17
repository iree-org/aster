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
// SelectLegalizePattern
//===----------------------------------------------------------------------===//

struct SelectLegalizePattern : public OpRewritePattern<lsir::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(lsir::SelectOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SCselectLegalizePattern
//===----------------------------------------------------------------------===//

// SOP2 allows at most one literal; materialize src0 into a register.
template <typename OpTy, int bitWidth>
struct SCselectLegalizePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override;
};

using SCselectB32LegalizePattern = SCselectLegalizePattern<SCselectB32, 32>;
using SCselectB64LegalizePattern = SCselectLegalizePattern<SCselectB64, 64>;

//===----------------------------------------------------------------------===//
// VCndmaskB32LegalizePattern
//===----------------------------------------------------------------------===//

struct VCndmaskB32LegalizePattern : public OpRewritePattern<VCndmaskB32> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(VCndmaskB32 op,
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
// SelectLegalizePattern
//===----------------------------------------------------------------------===//

LogicalResult
SelectLegalizePattern::matchAndRewrite(lsir::SelectOp op,
                                       PatternRewriter &rewriter) const {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();

  if (isa<VCCType>(op.getCondition().getType())) {
    // VOP2 src1 must be VGPR; VOP3 requires inline. Materialize into a VGPR.
    auto trueConst = getConstInt(op.getTrueValue(), 32);
    if (!trueConst || isInlineInt(*trueConst))
      return failure();
    Value out = createAllocation(rewriter, loc, getVGPR(ctx));
    Value movResult =
        VMovB32::create(rewriter, loc, out, op.getTrueValue()).getDst0Res();
    rewriter.modifyOpInPlace(
        op, [&]() { op.getTrueValueMutable().assign(movResult); });
    return success();
  }

  // SOP2 allows at most one literal; materialize true_value into an SGPR.
  auto trueConst = getConstInt(op.getTrueValue(), 32);
  auto falseConst = getConstInt(op.getFalseValue(), 32);
  if (!trueConst || !falseConst || isInlineInt(*trueConst) ||
      isInlineInt(*falseConst))
    return failure();
  Value out = createAllocation(rewriter, loc, getSGPR(ctx));
  Value movResult =
      SMovB32::create(rewriter, loc, getSGPR(ctx), out, op.getTrueValue())
          .getDst0Res();
  rewriter.modifyOpInPlace(
      op, [&]() { op.getTrueValueMutable().assign(movResult); });
  return success();
}

//===----------------------------------------------------------------------===//
// SCselectLegalizePattern
//===----------------------------------------------------------------------===//

template <typename OpTy, int bitWidth>
LogicalResult SCselectLegalizePattern<OpTy, bitWidth>::matchAndRewrite(
    OpTy op, PatternRewriter &rewriter) const {
  auto src0Const = getConstInt(op.getSrc0(), bitWidth);
  auto src1Const = getConstInt(op.getSrc1(), bitWidth);
  if (!src0Const || !src1Const || isInlineInt(*src0Const) ||
      isInlineInt(*src1Const))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();
  if constexpr (bitWidth == 32) {
    RegisterTypeInterface sgprTy = getSGPR(ctx);
    Value out = createAllocation(rewriter, loc, sgprTy);
    Value movResult =
        SMovB32::create(rewriter, loc, sgprTy, out, op.getSrc0()).getDst0Res();
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getSrc0Mutable().assign(movResult); });
    return success();
  }
  RegisterTypeInterface sgprTy = getSGPR(ctx, 2);
  Value out = createAllocation(rewriter, loc, sgprTy);
  Value movResult =
      SMovB64::create(rewriter, loc, sgprTy, out, op.getSrc0()).getDst0Res();
  rewriter.modifyOpInPlace(op,
                           [&]() { op.getSrc0Mutable().assign(movResult); });
  return success();
}

//===----------------------------------------------------------------------===//
// VCndmaskB32LegalizePattern
//===----------------------------------------------------------------------===//

LogicalResult
VCndmaskB32LegalizePattern::matchAndRewrite(VCndmaskB32 op,
                                            PatternRewriter &rewriter) const {
  // VOP2 src1 must be VGPR; VOP3 requires inline. Materialize into a VGPR.
  auto src1Const = getConstInt(op.getSrc1(), 32);
  if (!src1Const || isInlineInt(*src1Const))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();
  Value out = createAllocation(rewriter, loc, getVGPR(ctx));
  Value movResult =
      VMovB32::create(rewriter, loc, out, op.getSrc1()).getDst0Res();
  rewriter.modifyOpInPlace(op,
                           [&]() { op.getSrc1Mutable().assign(movResult); });
  return success();
}

//===----------------------------------------------------------------------===//
// LegalizeOperands pass
//===----------------------------------------------------------------------===//

void LegalizeOperands::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SelectLegalizePattern, SCselectB32LegalizePattern,
               SCselectB64LegalizePattern, VCndmaskB32LegalizePattern>(
      &getContext());
  if (failed(applyPatternsGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
    signalPassFailure();
}
