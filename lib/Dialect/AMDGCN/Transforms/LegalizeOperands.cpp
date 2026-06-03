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
// Legalizes constant operands of select instructions that the hardware encoding
// cannot satisfy.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
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

/// Return true if v is a non-inline-constant integer literal, i.e. an
/// arith.constant outside [-16, 64].
static bool isNonInlineLiteral(Value v) {
  if (auto c = getConstInt(v, 32))
    return !isInlineInt(*c);
  if (auto c = getConstInt(v, 64))
    return !isInlineInt(*c);
  return false;
}

template <typename MovTy, bool HasTypeArg>
static bool legalizeOperandWithMov(PatternRewriter &rewriter, Operation *op,
                                   OpOperand &operand,
                                   RegisterTypeInterface regTy) {
  if (!isNonInlineLiteral(operand.get()))
    return false;

  Location loc = op->getLoc();
  Value out = createAllocation(rewriter, loc, regTy);
  Value mov = [&]() -> Value {
    if constexpr (HasTypeArg)
      return MovTy::create(rewriter, loc, regTy, out, operand.get())
          .getDst0Res();
    return MovTy::create(rewriter, loc, out, operand.get()).getDst0Res();
  }();
  rewriter.modifyOpInPlace(op, [&]() { operand.set(mov); });
  return true;
}

namespace {

//===----------------------------------------------------------------------===//
// VCndmaskLegalizePattern
//===----------------------------------------------------------------------===//

/// From ISA guide 6.2.1: VALU "ADDC", "SUBB" and CNDMASK all implicitly use an
/// SGPR value (VCC), so these instructions cannot use an additional SGPR or
/// literal constant.
/// Note the difference between:
///  - inline constant: small values (integers [-16, 64], special floats) and
///    encoded directly in the instruction's 9-bit SRC.
///  - literal constant: full 32b value in the instruction stream
///    (encoding 255). These are explicitly forbidden with VCC.
struct VCndmaskLegalizePattern : OpRewritePattern<VCndmaskB32> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VCndmaskB32 op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    RegisterTypeInterface vgprTy = getVGPR(ctx);
    bool changed = legalizeOperandWithMov<VMovB32, /*HasTypeArg=*/false>(
        rewriter, op, op.getSrc0Mutable(), vgprTy);
    changed |= legalizeOperandWithMov<VMovB32, /*HasTypeArg=*/false>(
        rewriter, op, op.getSrc1Mutable(), vgprTy);
    return success(changed);
  }
};

//===----------------------------------------------------------------------===//
// SCselectLegalizePattern
//===----------------------------------------------------------------------===//

// s_cselect (SOP2) tolerates one literal.
template <typename OpTy, typename MovTy, int16_t RegCount>
struct SCselectLegalizePattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    OpOperand &src0 = op.getSrc0Mutable();
    if (!isNonInlineLiteral(src0.get()) ||
        !isNonInlineLiteral(op.getSrc1Mutable().get()))
      return failure();
    MLIRContext *ctx = rewriter.getContext();
    RegisterTypeInterface sgprTy = getSGPR(ctx, RegCount);
    bool changed = legalizeOperandWithMov<MovTy, /*HasTypeArg=*/true>(
        rewriter, op, src0, sgprTy);
    return success(changed);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// LegalizeOperands pass
//===----------------------------------------------------------------------===//

void LegalizeOperands::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<VCndmaskLegalizePattern,
               SCselectLegalizePattern<SCselectB32, SMovB32, 1>,
               SCselectLegalizePattern<SCselectB64, SMovB64, 2>>(&getContext());
  if (failed(applyPatternsGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
    signalPassFailure();
}
