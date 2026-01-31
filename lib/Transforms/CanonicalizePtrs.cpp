//===- CanonicalizePtrOps.cpp ---------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/TypeSize.h"

namespace mlir::aster {
#define GEN_PASS_DEF_CANONICALIZEPTROPS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// CanonicalizePtrOps pass
//===----------------------------------------------------------------------===//
struct CanonicalizePtrOps
    : public aster::impl::CanonicalizePtrOpsBase<CanonicalizePtrOps> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// ExpandTypeOffsetOpPattern
//===----------------------------------------------------------------------===//
struct ExpandTypeOffsetOpPattern : public OpRewritePattern<ptr::TypeOffsetOp> {
  ExpandTypeOffsetOpPattern(const DataLayout &dataLayout, MLIRContext *context,
                            PatternBenefit benefit = 1,
                            ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(ptr::TypeOffsetOp op,
                                PatternRewriter &rewriter) const override;

private:
  const DataLayout &dataLayout;
};

//===----------------------------------------------------------------------===//
// PtrAddZeroOpPattern
//===----------------------------------------------------------------------===//
struct PtrAddZeroOpPattern : public OpRewritePattern<ptr::PtrAddOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(ptr::PtrAddOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//
struct PtrAddOpPattern : public OpRewritePattern<ptr::PtrAddOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(ptr::PtrAddOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ExpandTypeOffsetOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ExpandTypeOffsetOpPattern::matchAndRewrite(ptr::TypeOffsetOp op,
                                           PatternRewriter &rewriter) const {
  Type elementType = op.getElementType();
  llvm::TypeSize typeSize = dataLayout.getTypeSize(elementType);
  if (!typeSize.isFixed())
    return failure();
  TypedAttr offsetAttr =
      op.getType().isIndex()
          ? rewriter.getIndexAttr(typeSize.getFixedValue())
          : rewriter.getIntegerAttr(op.getType(), typeSize.getFixedValue());
  auto nV = arith::ConstantOp::create(rewriter, op.getLoc(), offsetAttr);
  rewriter.replaceOp(op, nV.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

/// Combine PtrAddFlags from two ptr_add operations.
/// Returns the intersection (more conservative) of the two flags.
static ptr::PtrAddFlags combinePtrAddFlags(ptr::PtrAddFlags lhs,
                                           ptr::PtrAddFlags rhs) {
  // Return the minimum of the two flags (more restrictive).
  // none < nusw < nuw < inbounds in terms of guarantees.
  return static_cast<ptr::PtrAddFlags>(
      std::min(static_cast<uint32_t>(lhs), static_cast<uint32_t>(rhs)));
}

/// Compute IntegerOverflowFlags for arith::AddIOp based on combined
/// ptr::PtrAddFlags.
static arith::IntegerOverflowFlags
ptrFlagsToIntegerOverflowFlags(ptr::PtrAddFlags flags) {
  arith::IntegerOverflowFlags result = arith::IntegerOverflowFlags::none;
  // nuw (2) and inbounds (3) both imply no unsigned wrap.
  if (flags == ptr::PtrAddFlags::nuw || flags == ptr::PtrAddFlags::inbounds)
    result |= arith::IntegerOverflowFlags::nuw;
  // nusw (1) and inbounds (3) both imply no signed wrap.
  if (flags == ptr::PtrAddFlags::nusw || flags == ptr::PtrAddFlags::inbounds)
    result |= arith::IntegerOverflowFlags::nsw;
  return result;
}

LogicalResult
PtrAddOpPattern::matchAndRewrite(ptr::PtrAddOp op,
                                 PatternRewriter &rewriter) const {
  auto baseOp = op.getBase().getDefiningOp<ptr::PtrAddOp>();
  if (!baseOp || baseOp.getOffset().getType() != op.getOffset().getType())
    return failure();

  // Combine flags from both ptr_add operations.
  ptr::PtrAddFlags combinedFlags =
      combinePtrAddFlags(baseOp.getFlags(), op.getFlags());
  if (isa<IndexType>(op.getOffset().getType())) {
    affine::AffineApplyOp cOff = affine::makeComposedAffineApply(
        rewriter, op.getLoc(),
        rewriter.getAffineSymbolExpr(0) + rewriter.getAffineSymbolExpr(1),
        {baseOp.getOffset(), op.getOffset()});
    rewriter.replaceOpWithNewOp<ptr::PtrAddOp>(
        op, op.getType(), baseOp.getBase(), cOff, combinedFlags);
    return success();
  }
  arith::IntegerOverflowFlags overflowFlags =
      ptrFlagsToIntegerOverflowFlags(combinedFlags);

  arith::AddIOp cOff = rewriter.create<arith::AddIOp>(
      op.getLoc(), baseOp.getOffset(), op.getOffset(), overflowFlags);
  rewriter.replaceOpWithNewOp<ptr::PtrAddOp>(op, op.getType(), baseOp.getBase(),
                                             cOff, combinedFlags);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrAddZeroOpPattern::matchAndRewrite(ptr::PtrAddOp op,
                                     PatternRewriter &rewriter) const {
  if (Operation *cOp = op.getOffset().getDefiningOp();
      !cOp || !m_Zero().match(cOp))
    return failure();
  rewriter.replaceOp(op, op.getBase());
  return success();
}

//===----------------------------------------------------------------------===//
// CanonicalizePtrOps pass
//===----------------------------------------------------------------------===//

void CanonicalizePtrOps::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  DataLayoutAnalysis &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(op);
  MLIRContext *context = &getContext();
  {
    auto aD = context->getLoadedDialect<arith::ArithDialect>();
    aD->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op :
         context->getRegisteredOperationsByDialect(aD->getNamespace()))
      op.getCanonicalizationPatterns(patterns, context);
  }
  {
    auto aD = context->getLoadedDialect<affine::AffineDialect>();
    aD->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op :
         context->getRegisteredOperationsByDialect(aD->getNamespace()))
      op.getCanonicalizationPatterns(patterns, context);
  }
  patterns.add<PtrAddOpPattern, PtrAddZeroOpPattern>(context);
  patterns.add<ExpandTypeOffsetOpPattern>(dataLayout, context);
  if (failed(applyPatternsGreedily(
          op, FrozenRewritePatternSet(std::move(patterns)),
          GreedyRewriteConfig().setUseTopDownTraversal(true))))
    return signalPassFailure();
}
