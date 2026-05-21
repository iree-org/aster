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
//   layout.apply     -> affine.delinearize_index
//                       + affine.linearize_index_by_strides
//   layout.swizzle   -> arith bit ops (index_cast + shrui + andi + xori)
//
// The "disjoint" `affine.linearize_index` carries static "no internal alias"
// information that is useful for early analyses and canonicalizations.
// By the time we reach LowerLayoutToAffine we need to lower out to efficient
// low-level IR. `LinearizeIndexBoundedToStrides` rewrites "disjoint"
// `affine.linearize_index` to AffineLinearizeByStrideOp, which canonicalizes
// better around stride-1 patterns.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/Transforms/Passes.h"

#include "aster/Dialect/Layout/IR/LayoutAttrs.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"
#include "aster/Dialect/Layout/IR/LayoutOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

/// Product of all leaf values under an int-tuple shape attribute.
static int64_t productOfLeaves(Attribute attr) {
  SmallVector<int64_t> leaves;
  flattenToLeaves(attr, leaves);
  int64_t p = 1;
  for (int64_t v : leaves)
    p *= v;
  return p;
}

/// Recursively delinearize `coord` against the layout's shape tree.
static void recursiveDelinearize(PatternRewriter &rewriter, Location loc,
                                 Value coord, Attribute shapeAttr,
                                 SmallVectorImpl<Value> &leavesOut) {
  if (isa<IntegerAttr>(shapeAttr)) {
    leavesOut.push_back(coord);
    return;
  }
  auto arr = cast<ArrayAttr>(shapeAttr);
  if (arr.size() == 1) {
    recursiveDelinearize(rewriter, loc, coord, arr[0], leavesOut);
    return;
  }
  SmallVector<int64_t> childProducts;
  childProducts.reserve(arr.size());
  for (Attribute child : arr)
    childProducts.push_back(productOfLeaves(child));
  auto delinOp = affine::AffineDelinearizeIndexOp::create(rewriter, loc, coord,
                                                          childProducts);
  for (auto [childCoord, child] : llvm::zip(delinOp.getResults(), arr))
    recursiveDelinearize(rewriter, loc, childCoord, child, leavesOut);
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower layout.apply. Verifier guarantees one of two arities:
///   1. 1 coord (linear): recursive delinearize against the layout's shape
///      tree (one affine.delinearize_index per nested mode against that
///      mode's children's products), then dot with flat strides.
///   2. flat_rank(layout) coords (decomposed): dot directly with flat strides.
struct LowerApplyPattern : public OpRewritePattern<ApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();
    Location loc = op.getLoc();
    ValueRange coords = op.getCoords();

    SmallVector<int64_t> flatShape, flatStride;
    for (auto s : layout.getShape())
      flattenToLeaves(s, flatShape);
    for (auto d : layout.getStride())
      flattenToLeaves(d, flatStride);
    assert(flatShape.size() == flatStride.size());
    size_t flatRank = flatShape.size();

    SmallVector<Value> dotCoords;
    if (coords.size() == flatRank) {
      // Decomposed form: just dot.
      dotCoords.assign(coords.begin(), coords.end());
    } else {
      // Linear form: recursive delinearize against the shape tree.
      assert(coords.size() == 1 && "verifier should reject other arities");
      recursiveDelinearize(rewriter, loc, coords[0], layout.getShape(),
                           dotCoords);
    }

    // Prefer `linearize_index` bounded form when strides are
    // suffix_product(shape): it preserves the static range bound at the type
    // level for downstream analyses.
    Value result;
    if (computeSuffixProduct(flatShape) == ArrayRef<int64_t>(flatStride)) {
      result = affine::AffineLinearizeIndexOp::create(
          rewriter, loc, dotCoords, flatShape, /*disjoint=*/true);
    } else {
      result = affine::AffineLinearizeIndexByStridesOp::create(
          rewriter, loc, dotCoords, flatStride);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrite `affine.linearize_index disjoint [c0..cN] by (B0..BN)` (outer-
/// bounded, fully static basis) to `affine.linearize_index_by_strides
/// [c0..cN] by suffix_product(B0..BN)`.
///
/// Both forms compute the same arithmetic. The bounded form encodes a static
/// range bound at the type level, but downstream canonicalization patterns
/// simplify the strides form more aggressively (drop unit-stride dimensions,
/// fuse with consuming `affine.apply`), which produces tighter address
/// arithmetic during scheduling and codegen.
struct LinearizeIndexBoundedToStrides
    : public OpRewritePattern<affine::AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasOuterBound())
      return rewriter.notifyMatchFailure(op, "no outer bound");
    if (!op.getDynamicBasis().empty())
      return rewriter.notifyMatchFailure(op, "dynamic basis");

    ArrayRef<int64_t> basis = op.getStaticBasis();
    SmallVector<int64_t> strides = computeSuffixProduct(basis);
    rewriter.replaceOpWithNewOp<affine::AffineLinearizeIndexByStridesOp>(
        op, op.getMultiIndex(), strides);
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
// ThreadValueOffsetsToApply
//===----------------------------------------------------------------------===//

/// Based offset rewritten to `layout.apply[%tid], #thread_layout`, then one
/// `affine.apply (d0 + vOff)` per non-zero offset.
struct ThreadValueOffsetsToApply : OpRewritePattern<ThreadValueOffsetsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ThreadValueOffsetsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    LayoutAttr val = op.getValueLayout();
    int64_t n = val.getSize();

    Value baseOffset =
        ApplyOp::create(rewriter, loc, rewriter.getIndexType(),
                        ValueRange{op.getTid()}, op.getThreadLayout())
            .getResult();

    SmallVector<Value> results;
    results.reserve(n);
    results.push_back(baseOffset);
    for (int64_t v = 1; v < n; ++v) {
      int64_t vOff = val.evaluate(v);
      auto d0 = getAffineDimExpr(0, ctx);
      auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, d0 + vOff);
      results.push_back(affine::AffineApplyOp::create(rewriter, loc, map,
                                                      ValueRange{baseOffset})
                            .getResult());
    }
    rewriter.replaceOp(op, results);
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
    patterns.add<ThreadValueOffsetsToApply, LowerApplyPattern,
                 LinearizeIndexBoundedToStrides, LowerSwizzlePattern>(
        &getContext());

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
