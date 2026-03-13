//===- ExpandAffineApply.cpp - Expand affine.apply to n-ary ops ----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_EXPANDAFFINEAPPLY
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {

/// Create a composed/folded affine.apply for the given single-result map and
/// its SSA operands. Constant results are materialized as arith.constant.
static Value applyAndFold(IRRewriter &rewriter, Location loc, AffineMap map,
                          ArrayRef<Value> mapOperands) {
  SmallVector<OpFoldResult> ofrs(mapOperands.begin(), mapOperands.end());
  OpFoldResult ofr =
      affine::makeComposedFoldedAffineApply(rewriter, loc, map, ofrs);
  if (Value val = dyn_cast<Value>(ofr))
    return val;
  return arith::ConstantIndexOp::create(
             rewriter, loc, cast<IntegerAttr>(cast<Attribute>(ofr)).getInt())
      .getResult();
}

/// Flatten a binary op chain of the given kind into a list of leaf
/// sub-expressions by recursively splitting LHS and RHS.
static void collectChain(AffineExpr expr, AffineExprKind kind,
                         SmallVectorImpl<AffineExpr> &chain) {
  auto binOp = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binOp || binOp.getKind() != kind) {
    chain.push_back(expr);
    return;
  }
  collectChain(binOp.getLHS(), kind, chain);
  collectChain(binOp.getRHS(), kind, chain);
}

/// Recursively lower an AffineExpr to a Value, inserting ops before the
/// current insertion point. Add and mul chains are collected into n-ary
/// aster_utils.addi / aster_utils.muli. All other binary expressions have
/// their operands recursively lowered and then re-emitted as a new
/// (composed/folded) affine.apply over the two lowered SSA symbol operands.
static Value lowerAffineExpr(IRRewriter &rewriter, Location loc,
                             AffineExpr expr, ValueRange operands,
                             unsigned numDims) {
  if (auto cst = dyn_cast<AffineConstantExpr>(expr))
    return arith::ConstantIndexOp::create(rewriter, loc, cst.getValue())
        .getResult();

  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return operands[dim.getPosition()];

  if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
    return operands[numDims + sym.getPosition()];

  auto binOp = cast<AffineBinaryOpExpr>(expr);
  AffineExprKind kind = binOp.getKind();

  if (kind == AffineExprKind::Add || kind == AffineExprKind::Mul) {
    SmallVector<AffineExpr> chain;
    collectChain(expr, kind, chain);
    assert(chain.size() >= 2 &&
           "binary op must yield at least two chain elements");

    SmallVector<Value> values;
    values.reserve(chain.size());
    for (AffineExpr e : chain)
      values.push_back(lowerAffineExpr(rewriter, loc, e, operands, numDims));

    Type indexType = rewriter.getIndexType();
    if (kind == AffineExprKind::Add)
      return AddiOp::create(rewriter, loc, indexType, values);
    return MuliOp::create(rewriter, loc, indexType, values);
  }

  // For mod, floordiv, and ceildiv: recursively lower both operands, then
  // wrap them in a new (composed/folded) affine.apply that applies only the
  // single binary op over two fresh symbol operands.
  Value lhs = lowerAffineExpr(rewriter, loc, binOp.getLHS(), operands, numDims);
  Value rhs = lowerAffineExpr(rewriter, loc, binOp.getRHS(), operands, numDims);

  MLIRContext *ctx = rewriter.getContext();
  AffineExpr s0 = getAffineSymbolExpr(0, ctx);
  AffineExpr s1 = getAffineSymbolExpr(1, ctx);

  AffineExpr mapExpr;
  switch (kind) {
  case AffineExprKind::Mod:
    mapExpr = s0 % s1;
    break;
  case AffineExprKind::FloorDiv:
    mapExpr = s0.floorDiv(s1);
    break;
  case AffineExprKind::CeilDiv:
    mapExpr = s0.ceilDiv(s1);
    break;
  default:
    llvm_unreachable("unhandled AffineExprKind");
  }

  AffineMap map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/2, mapExpr, ctx);
  return applyAndFold(rewriter, loc, map, {lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ExpandAffineApply pass
//===----------------------------------------------------------------------===//

struct ExpandAffineApply
    : public aster_utils::impl::ExpandAffineApplyBase<ExpandAffineApply> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](affine::AffineApplyOp op) {
      rewriter.setInsertionPoint(op);
      AffineMap map = op.getAffineMap();
      assert(map.getNumResults() == 1 &&
             "multi-result affine.apply not supported");
      Value result = lowerAffineExpr(rewriter, op.getLoc(), map.getResult(0),
                                     op.getOperands(), map.getNumDims());
      rewriter.replaceAllUsesWith(op.getResult(), result);
      rewriter.eraseOp(op);
    });
  }
};

} // namespace
