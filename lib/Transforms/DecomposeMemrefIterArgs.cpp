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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
/// These are iter_args where stores and loads both target the same block arg
/// at constant indices. After self-forwarding, erases dead stores on both the
/// block arg and the for-op result.
static void forwardStaticAllocaStores(scf::ForOp forOp) {
  int64_t numArgs = forOp.getNumRegionIterArgs();
  IRRewriter rewriter(forOp.getContext());

  for (int64_t i = 0; i < numArgs; ++i) {
    Value init = forOp.getInitArgs()[i];
    auto memrefType = dyn_cast<MemRefType>(init.getType());
    if (!memrefType || memrefType.getRank() != 1 ||
        !memrefType.hasStaticShape())
      continue;

    int64_t numElements = memrefType.getShape()[0];
    BlockArgument ba = forOp.getRegionIterArgs()[i];

    // Self-forward: stores and loads target the same block arg.
    if (failed(forwardConstantIndexStores(rewriter, ba, ba, numElements))) {
      LDBG() << "  SKIP iter_arg #" << i << ": self-forwarding failed";
      continue;
    }
    LDBG() << "  Forwarded iter_arg #" << i << " (" << numElements
           << " elements)";

    // Best-effort post-loop cleanup: the primary in-loop forwarding already
    // succeeded above. These may fail if the post-loop result or init don't
    // match the expected pattern (e.g., no stores, or still has loads) -- that
    // is expected and not a pass failure.
    Value result = forOp.getResult(i);
    (void)forwardConstantIndexStores(rewriter, result, result, numElements);
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
}
