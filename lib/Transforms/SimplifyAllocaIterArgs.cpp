//===- SimplifyAllocaIterArgs.cpp - Simplify alloca iter_args -------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pipeliner creates BOTH a static alloca and its dynamic cast as separate
// iter_args:
//
//   %alloca = memref.alloca() : memref<4xT>
//   %cast = memref.cast %alloca : ... to memref<?xT>
//   scf.for iter_args(%argS = %alloca, %argD = %cast) {
//     store %val, %argS[i]; %v = load %argD[i]  // aliased access
//     yield %new_alloca, %new_cast
//   }
//
// This pass eliminates paired alloca + cast iter_args and is analogous to
// the MLIR upstream ForOpTensorCastFolder, for memref.cast.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-simplify-alloca-iter-args"

namespace mlir::aster {
#define GEN_PASS_DEF_SIMPLIFYALLOCAITERARGS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {

/// Fold memref.cast on scf.for iter_args. When an iter_arg's init value is
/// a memref.cast from a more-static type, replace the iter_arg with the
/// source type and insert casts in the body and after the loop.
struct ForOpMemRefCastFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    for (auto it : llvm::zip(op.getInitArgsMutable(), op.getResults())) {
      OpOperand &iterOpOperand = std::get<0>(it);
      auto incomingCast = iterOpOperand.get().getDefiningOp<memref::CastOp>();
      if (!incomingCast)
        continue;

      auto srcType = cast<MemRefType>(incomingCast.getSource().getType());
      auto dstType = cast<MemRefType>(incomingCast.getType());
      if (srcType == dstType)
        continue;

      // Source must have strictly more static info than dest (upstream helper).
      if (!memref::CastOp::canFoldIntoConsumerOp(incomingCast))
        continue;

      LDBG() << "Fold memref.cast on iter_arg " << srcType << " -> " << dstType;

      rewriter.replaceOp(
          op, scf::replaceAndCastForOpIterArg(
                  rewriter, op, iterOpOperand, incomingCast.getSource(),
                  [](OpBuilder &b, Location loc, Type type, Value source) {
                    return memref::CastOp::create(b, loc, type, source);
                  }));
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// SimplifyAllocaIterArgs pass
//===----------------------------------------------------------------------===//

struct SimplifyAllocaIterArgs
    : public mlir::aster::impl::SimplifyAllocaIterArgsBase<
          SimplifyAllocaIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void SimplifyAllocaIterArgs::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ForOpMemRefCastFolder>(&getContext());
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      patterns, scf::ForOp::getOperationName());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitWarning("greedy rewrite failed");
    return signalPassFailure();
  }
}
