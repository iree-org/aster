//===- PostRegAllocLegalization.cpp - Legalize ops after register alloc ===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/InstOpInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "amdgcn-post-reg-alloc-legalization"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_POSTREGALLOCLEGALIZATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// PostRegAllocLegalization pass
//===----------------------------------------------------------------------===//
struct PostRegAllocLegalization
    : public amdgcn::impl::PostRegAllocLegalizationBase<
          PostRegAllocLegalization> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

struct CopyExpansionPattern : public OpRewritePattern<lsir::CopyOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(lsir::CopyOp op,
                                PatternRewriter &rewriter) const override;
};

struct MovInstOpPattern : public OpInterfaceRewritePattern<MovInstOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(MovInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Returns true if the value needs allocation (i.e. is not yet an allocated
// physical register). Also defined in RegisterColoring.cpp; kept local to
// avoid coupling the two passes.
static bool needsAllocation(Value node) {
  auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(node.getType());
  return regTy && !regTy.hasAllocatedSemantics();
}

//===----------------------------------------------------------------------===//
// CopyExpansionPattern
//===----------------------------------------------------------------------===//

LogicalResult
CopyExpansionPattern::matchAndRewrite(lsir::CopyOp op,
                                      PatternRewriter &rewriter) const {
  // Self-copies were already removed by RegisterColoring. Remaining copies
  // with fully allocated operands are expanded to hardware mov instructions.

  // Value-semantics copies with live results must have been eliminated by
  // ToRegisterSemantics before register coloring runs.
  if (op.getTargetRes() && !op.getTargetRes().use_empty())
    return failure();

  // Get the source allocas. Bail out if the allocas are missing or need
  // allocation.
  FailureOr<ValueRange> srcAlloc = getAllocasOrFailure(op.getSource());
  if (failed(srcAlloc) ||
      llvm::any_of(*srcAlloc, [](Value v) { return needsAllocation(v); }))
    return failure();

  // Get the target allocas. Bail out if the allocas are missing or need
  // allocation.
  FailureOr<ValueRange> tgtAlloc = getAllocasOrFailure(op.getTarget());
  if (failed(tgtAlloc) ||
      llvm::any_of(*tgtAlloc, [](Value v) { return needsAllocation(v); }))
    return failure();

  assert(srcAlloc->size() == tgtAlloc->size() &&
         "source and target allocas must have the same size");
  assert(srcAlloc->size() > 0 &&
         "source and target allocas must have at least one alloca");

  auto srcTy = dyn_cast<AMDGCNRegisterTypeInterface>(op.getSource().getType());
  auto tgtTy = dyn_cast<AMDGCNRegisterTypeInterface>(op.getTarget().getType());

  // Bail if the source or target is not an AMDGCN register type.
  if (!srcTy || !tgtTy)
    return failure();

  // Bail if the copy cannot be performed.
  if (srcTy.getRegisterKind() != RegisterKind::SGPR &&
      tgtTy.getRegisterKind() == RegisterKind::SGPR) {
    return rewriter.notifyMatchFailure(
        op, "cannot copy between non-sgpr type to an sgpr type");
  }
  if (!llvm::is_contained(
          {RegisterKind::SGPR, RegisterKind::VGPR, RegisterKind::AGPR},
          srcTy.getRegisterKind()) ||
      !llvm::is_contained(
          {RegisterKind::SGPR, RegisterKind::VGPR, RegisterKind::AGPR},
          tgtTy.getRegisterKind())) {
    return rewriter.notifyMatchFailure(
        op, "cannot copy if data type is not SGPR, VGPR, or AGPR");
  }
  // AGPR copies: only AGPR->AGPR is supported (via v_accvgpr_mov_b32).
  if (tgtTy.getRegisterKind() == RegisterKind::AGPR &&
      srcTy.getRegisterKind() != RegisterKind::AGPR) {
    return rewriter.notifyMatchFailure(
        op,
        "cannot copy non-AGPR to AGPR (use v_accvgpr_write_b32 explicitly)");
  }
  if (srcTy.getRegisterKind() == RegisterKind::AGPR &&
      tgtTy.getRegisterKind() != RegisterKind::AGPR) {
    return rewriter.notifyMatchFailure(
        op, "cannot copy AGPR to non-AGPR (use v_accvgpr_read_b32 explicitly)");
  }

  auto copyReg = [&](Value src, Value tgt) {
    if (tgtTy.getRegisterKind() == RegisterKind::SGPR) {
      SMovB32::create(rewriter, tgt.getLoc(), tgt, src);
      return;
    }
    if (tgtTy.getRegisterKind() == RegisterKind::AGPR) {
      VAccvgprMovB32::create(rewriter, tgt.getLoc(), tgt, src);
      return;
    }
    VMovB32::create(rewriter, tgt.getLoc(), tgt, src);
  };

  // Emit one hardware mov per register pair.
  for (auto [src, tgt] : llvm::zip_equal(*srcAlloc, *tgtAlloc))
    copyReg(src, tgt);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// MovInstOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
MovInstOpPattern::matchAndRewrite(MovInstOpInterface op,
                                  PatternRewriter &rewriter) const {
  return aster::detail::canonicalizeMovInstImpl(op, rewriter);
}

//===----------------------------------------------------------------------===//
// PostRegAllocLegalization pass
//===----------------------------------------------------------------------===//

void PostRegAllocLegalization::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<CopyExpansionPattern>(&getContext(), /*benefit=*/1);
  patterns.add<MovInstOpPattern>(&getContext(), /*benefit=*/2);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPatternsGreedily(getOperation(), frozenPatterns)))
    return signalPassFailure();
}
