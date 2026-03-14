//===- RaiseToAffine.cpp - Raise addi/muli to affine.apply ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_RAISETOAFFINE
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {

static AffineMap buildNaryMap(MLIRContext *ctx, unsigned n, bool isAdd) {
  AffineExpr expr = getAffineSymbolExpr(0, ctx);
  for (unsigned i = 1; i < n; ++i) {
    AffineExpr next = getAffineSymbolExpr(i, ctx);
    expr = isAdd ? (expr + next) : (expr * next);
  }
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/n, expr, ctx);
}

struct AddiToAffineApply : public OpRewritePattern<AddiOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddiOp op,
                                PatternRewriter &rewriter) const override {
    ValueRange inputs = op.getInputs();
    AffineMap map = buildNaryMap(rewriter.getContext(), inputs.size(), true);
    SmallVector<OpFoldResult> ofrs(inputs.begin(), inputs.end());
    rewriter.replaceOp(
        op, affine::makeComposedAffineApply(rewriter, op.getLoc(), map, ofrs));
    return success();
  }
};

struct MuliToAffineApply : public OpRewritePattern<MuliOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MuliOp op,
                                PatternRewriter &rewriter) const override {
    ValueRange inputs = op.getInputs();
    AffineMap map = buildNaryMap(rewriter.getContext(), inputs.size(), false);
    SmallVector<OpFoldResult> ofrs(inputs.begin(), inputs.end());
    rewriter.replaceOp(
        op, affine::makeComposedAffineApply(rewriter, op.getLoc(), map, ofrs));
    return success();
  }
};

struct RaiseToAffine
    : public aster_utils::impl::RaiseToAffineBase<RaiseToAffine> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AddiToAffineApply, MuliToAffineApply>(ctx);
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
