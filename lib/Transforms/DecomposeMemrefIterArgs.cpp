//===- DecomposeMemrefIterArgs.cpp - Decompose memref iter_args -----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forwards stores-to-loads for static alloca iter_args (memref<NxT>) where
// stores and loads target the same block arg at constant indices. After
// forwarding, erases dead stores so canonicalize can remove unused iter_args.
//
// First, use aster-simplify-alloca-iter-args first to fold casts and dedup
// iter_args, then this pass handles the forwarding.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/MemRefUtils.h"
#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-decompose-memref-iter-args"

namespace mlir::aster {
#define GEN_PASS_DEF_DECOMPOSEMEMREFITERARGS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {

/// Forward stores-to-loads for static alloca iter_args (memref<NxT>).
///
/// Two patterns are handled:
///
/// Pattern 1 - Self-forwarding: stores and loads both target the same block
/// arg at constant indices (typical after SimplifyAllocaIterArgs dedup).
///
/// Pattern 2 - Cross-iteration forwarding: stores go to a NEW alloca that is
/// yielded, loads come from the BLOCK ARG (previous iteration's yield).
/// This is the natural pattern from pipelined loops with buffer APIs:
///   scf.for iter_args(%ba = %alloca) {
///     %new = memref.alloca()
///     store %v, %new[0]       // stores to yield value
///     load %ba[0]             // loads from block arg
///     yield %new
///   }
static void forwardStaticAllocaStores(scf::ForOp forOp) {
  int64_t numArgs = forOp.getNumRegionIterArgs();
  IRRewriter rewriter(forOp.getContext());
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (int64_t i = 0; i < numArgs; ++i) {
    Value init = forOp.getInitArgs()[i];
    auto memrefType = dyn_cast<MemRefType>(init.getType());
    if (!memrefType || memrefType.getRank() != 1 ||
        !memrefType.hasStaticShape())
      continue;

    int64_t numElements = memrefType.getShape()[0];
    BlockArgument ba = forOp.getRegionIterArgs()[i];

    // Try self-forwarding first (Pattern 1).
    if (succeeded(forwardConstantIndexStores(rewriter, ba, ba, numElements))) {
      LDBG() << "  Self-forwarded iter_arg #" << i << " (" << numElements
             << " elements)";
    } else {
      // Try cross-iteration forwarding (Pattern 2): stores to yield value,
      // loads from block arg.
      Value yieldVal = yieldOp.getOperand(i);
      if (yieldVal == ba) {
        LDBG() << "  SKIP iter_arg #" << i << ": yield == ba, no forwarding";
        continue;
      }
      if (failed(forwardConstantIndexStores(rewriter, yieldVal, ba,
                                            numElements))) {
        LDBG() << "  SKIP iter_arg #" << i << ": cross-iteration failed";
        continue;
      }
      LDBG() << "  Cross-forwarded iter_arg #" << i << " (" << numElements
             << " elements)";
    }

    // Best-effort cleanup: erase dead stores on the block arg (loads were
    // forwarded or didn't exist), the for-op result, and the init alloca.
    // These may fail if the pattern doesn't match -- that is expected.
    (void)eraseDeadMemrefStores(rewriter, ba);
    Value result = forOp.getResult(i);
    // Try self-forwarding on result, then cross-forwarding from init to
    // result (init stores dominate post-loop result loads in the same block).
    (void)forwardConstantIndexStores(rewriter, result, result, numElements);
    (void)forwardConstantIndexStores(rewriter, init, result, numElements);
    (void)eraseDeadMemrefStores(rewriter, result);
    (void)eraseDeadMemrefStores(rewriter, init);
  }
}

//===----------------------------------------------------------------------===//
// DecomposeMemrefIterArgs pass
//===----------------------------------------------------------------------===//

struct DecomposeMemrefIterArgs
    : public mlir::aster::impl::DecomposeMemrefIterArgsBase<
          DecomposeMemrefIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void DecomposeMemrefIterArgs::runOnOperation() {
  Operation *rootOp = getOperation();
  SmallVector<scf::ForOp> forOps;
  rootOp->walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (scf::ForOp forOp : forOps)
    forwardStaticAllocaStores(forOp);

  // Clean up: remove unused iter_args, dead allocas, dead constants.
  MLIRContext *ctx = rootOp->getContext();
  RewritePatternSet patterns(ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  memref::AllocaOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(rootOp, std::move(patterns));
}
