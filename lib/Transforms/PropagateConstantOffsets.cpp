//===- PropagateConstantOffsets.cpp
//----------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propagates constant offsets through affine.apply and arith operations.
//
// Two key patterns:
//   1. ExtractConstantApplyOffset: Extracts the constant term from an
//      affine.apply map into an explicit arith.addi.
//        affine.apply<expr + C>  -->  arith.addi(affine.apply<expr>, C)
//
//   2. FoldApplySymbolOrDimSum: Absorbs arith.addi(x, C) operands back
//      into the affine map, composing constant offsets.
//        affine.apply<f(d0)>(x + C)  -->  affine.apply<f(d0 + C)>(x)
//
// Together with canonicalization these create a fixed-point iteration that
// separates constant address offsets from dynamic index computations.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
#define GEN_PASS_DEF_PROPAGATECONSTANTOFFSETS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Return the constant RHS of an arith op (AddIOp or MulIOp) if the op
/// has the nsw overflow flag set. Otherwise return std::nullopt.
template <typename OpTy>
static std::optional<int64_t> getValueConstantRhs(Value v) {
  auto op = v.getDefiningOp<OpTy>();
  if (!op)
    return std::nullopt;
  if (!bitEnumContainsAll(op.getOverflowFlags(),
                          arith::IntegerOverflowFlags::nsw))
    return std::nullopt;
  APInt value;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&value)))
    return std::nullopt;
  return value.getSExtValue();
}

/// Extract the constant offset from an affine.apply map.
///
///   affine.apply affine_map<expr + C>(operands)
/// becomes:
///   %apply = affine.apply affine_map<expr>(operands)
///   arith.addi %apply, C overflow<nsw>
///
struct ExtractConstantApplyOffset : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.getAffineMap();
    assert(map.getNumResults() == 1 && "expected single result map");

    // Simplify to normalize additive constant to the RHS.
    AffineMap simplified = simplifyAffineMap(map);
    AffineExpr expr = simplified.getResult(0);

    // Check for expr + C pattern.
    auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!addExpr || addExpr.getKind() != AffineExprKind::Add)
      return failure();
    auto constExpr = dyn_cast<AffineConstantExpr>(addExpr.getRHS());
    if (!constExpr || constExpr.getValue() == 0)
      return failure();

    // Create new affine.apply without the constant offset.
    AffineMap newMap = AffineMap::get(
        simplified.getNumDims(), simplified.getNumSymbols(), addExpr.getLHS());
    Value newApply =
        rewriter.create<AffineApplyOp>(op.getLoc(), newMap, op.getOperands());
    Value offset = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), constExpr.getValue());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op, newApply, offset, arith::IntegerOverflowFlags::nsw);
    return success();
  }
};

/// Fold arith.addi(x, C) operands into the affine map.
///
///   affine.apply affine_map<f(d0)>(arith.addi(x, C))
/// becomes:
///   affine.apply affine_map<f(d0 + C)>(x)
///
struct FoldApplySymbolOrDimSum : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.getAffineMap();
    unsigned numDims = map.getNumDims();
    unsigned numSymbols = map.getNumSymbols();

    SmallVector<Value> newOperands(op.getOperands());
    SmallVector<AffineExpr> dimReplacements;
    SmallVector<AffineExpr> symReplacements;
    bool changed = false;

    // Check dims.
    for (unsigned i = 0; i < numDims; ++i) {
      AffineExpr dimExpr = rewriter.getAffineDimExpr(i);
      auto addConst = getValueConstantRhs<arith::AddIOp>(newOperands[i]);
      if (addConst) {
        // Replace operand with the LHS of the add, fold constant into map.
        newOperands[i] = newOperands[i].getDefiningOp<arith::AddIOp>().getLhs();
        dimReplacements.push_back(dimExpr +
                                  rewriter.getAffineConstantExpr(*addConst));
        changed = true;
      } else {
        dimReplacements.push_back(dimExpr);
      }
    }

    // Check symbols.
    for (unsigned i = 0; i < numSymbols; ++i) {
      AffineExpr symExpr = rewriter.getAffineSymbolExpr(i);
      auto addConst =
          getValueConstantRhs<arith::AddIOp>(newOperands[numDims + i]);
      if (addConst) {
        newOperands[numDims + i] =
            newOperands[numDims + i].getDefiningOp<arith::AddIOp>().getLhs();
        symReplacements.push_back(symExpr +
                                  rewriter.getAffineConstantExpr(*addConst));
        changed = true;
      } else {
        symReplacements.push_back(symExpr);
      }
    }

    if (!changed)
      return failure();

    AffineMap newMap = map.replaceDimsAndSymbols(
        dimReplacements, symReplacements, numDims, numSymbols);
    rewriter.modifyOpInPlace(op, [&]() {
      op.setMap(newMap);
      op->setOperands(newOperands);
    });
    return success();
  }
};

struct PropagateConstantOffsets
    : public aster::impl::PropagateConstantOffsetsBase<
          PropagateConstantOffsets> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExtractConstantApplyOffset, FoldApplySymbolOrDimSum>(
        &getContext());
    // Include canonicalization to compose affine.apply chains and merge
    // constant adds, creating new rewrite opportunities.
    AffineApplyOp::getCanonicalizationPatterns(patterns, &getContext());
    arith::AddIOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
