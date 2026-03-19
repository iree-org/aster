//===- LowerLayoutToAffine.cpp - Lower layout ops to affine + arith ------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers layout dialect ops:
//   layout.linearize -> affine.delinearize_index
//                       + affine.linearize_index_by_strides
//   layout.swizzle   -> arith bit ops (index_cast + shrui + andi + xori)
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/Transforms/Passes.h"

#include "aster/Dialect/Layout/IR/LayoutAttrs.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"
#include "aster/Dialect/Layout/IR/LayoutOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster::layout {
#define GEN_PASS_DEF_LOWERLAYOUTTOAFFINE
#include "aster/Dialect/Layout/Transforms/Passes.h.inc"
} // namespace mlir::aster::layout

using namespace mlir;
using namespace mlir::aster::layout;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Recursively flatten an int-tuple element to leaf values.
static void flattenToLeaves(Attribute attr, SmallVectorImpl<int64_t> &out) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out.push_back(intAttr.getInt());
    return;
  }
  for (auto elem : cast<ArrayAttr>(attr))
    flattenToLeaves(elem, out);
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower layout.linearize to affine.delinearize_index (coordinate extraction)
/// + affine.linearize_index_by_strides (explicit stride dot product).
struct LowerLinearizePattern : public OpRewritePattern<LinearizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearizeOp op,
                                PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();
    Location loc = op.getLoc();
    Value coord = op.getCoord();

    SmallVector<int64_t> flatShape, flatStride;
    for (auto s : layout.getShape())
      flattenToLeaves(s, flatShape);
    for (auto d : layout.getStride())
      flattenToLeaves(d, flatStride);

    assert(flatShape.size() == flatStride.size());

    // Step 1: Delinearize the flat coordinate into multi-index by shape.
    auto delinOp = affine::AffineDelinearizeIndexOp::create(rewriter, loc,
                                                            coord, flatShape);

    // Step 2: Dot product with explicit strides.
    Value result = affine::AffineLinearizeIndexByStridesOp::create(
        rewriter, loc, delinOp.getResults(), flatStride);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerSwizzlePattern : public OpRewritePattern<SwizzleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwizzleOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value offset = op.getOffset();
    int64_t bits = op.getBits();
    int64_t base = op.getBase();
    int64_t shift = op.getShift();

    // result = offset ^ ((offset >> shift) & mask)
    // Work in i32 because ASTER's backend doesn't support bitwise ops on index.
    int64_t maskVal = ((1LL << bits) - 1) << base;
    auto i32Ty = rewriter.getI32Type();

    Value off32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, offset);
    Value shiftConst = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(shift));
    Value maskConst = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(maskVal));
    Value shifted = arith::ShRUIOp::create(rewriter, loc, off32, shiftConst);
    Value masked = arith::AndIOp::create(rewriter, loc, shifted, maskConst);
    Value xored = arith::XOrIOp::create(rewriter, loc, off32, masked);
    Value result = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), xored);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct LowerLayoutToAffinePass
    : public mlir::aster::layout::impl::LowerLayoutToAffineBase<
          LowerLayoutToAffinePass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLinearizePattern, LowerSwizzlePattern>(&getContext());

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig()
                .setUseTopDownTraversal(true)
                .setRegionSimplificationLevel(
                    GreedySimplifyRegionLevel::Disabled)))) {
      signalPassFailure();
    }
  }
};

} // namespace
