//===- PromoteLoopCarriedMemrefs.cpp - Promote memrefs to iter_args -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promotes scalar memref allocas with loop-carried load/store patterns to
// scf.for iter_args. After constexpr expansion + SROA + mem2reg, accumulator
// allocas remain as individual scalar memrefs:
//
//   memref.store %init, %alloca[] : memref<T>
//   scf.for %k = ... {
//     %old = memref.load %alloca[] : memref<T>
//     %new = compute(%old)
//     memref.store %new, %alloca[] : memref<T>
//   }
//   %final = memref.load %alloca[] : memref<T>
//
// This pass converts them to scf.for iter_args:
//
//   %final = scf.for %k = ... iter_args(%acc = %init) {
//     %new = compute(%acc)
//     scf.yield %new
//   }
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/MemRefUtils.h"
#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
#define GEN_PASS_DEF_PROMOTELOOPCARRIEDMEMREFS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {

/// Describes a single scalar alloca with a loop-carried load/store pattern.
struct LoopCarriedAlloca {
  memref::AllocaOp allocaOp;
  /// The store before the loop that sets the initial value.
  memref::StoreOp preLoopStore;
  /// The load inside the loop body that reads the current value.
  memref::LoadOp inLoopLoad;
  /// The store inside the loop body that writes the new value.
  memref::StoreOp inLoopStore;
  /// The load after the loop that reads the final value.
  memref::LoadOp postLoopLoad;
  /// The enclosing scf.for.
  scf::ForOp forOp;
};

/// Check if an operation is directly inside the body of the given scf.for
/// (not nested inside an scf.if or other op with regions).
static bool isInsideLoop(Operation *op, scf::ForOp forOp) {
  return op->getBlock() == forOp.getBody();
}

/// Check if an operation is strictly before the given scf.for (in the same
/// block).
static bool isBeforeLoop(Operation *op, scf::ForOp forOp) {
  return op->getBlock() == forOp->getBlock() && op->isBeforeInBlock(forOp);
}

/// Check if an operation is strictly after the given scf.for (in the same
/// block).
static bool isAfterLoop(Operation *op, scf::ForOp forOp) {
  return op->getBlock() == forOp->getBlock() && forOp->isBeforeInBlock(op);
}

/// Try to classify a memref.alloca as a loop-carried accumulator pattern.
/// Returns std::nullopt if the alloca does not match the pattern.
static std::optional<LoopCarriedAlloca>
classifyAlloca(memref::AllocaOp allocaOp) {
  auto memrefType = allocaOp.getType();
  if (!isScalarMemRef(memrefType))
    return std::nullopt;

  // Collect all users and classify them.
  memref::StoreOp preLoopStore;
  memref::LoadOp inLoopLoad;
  memref::StoreOp inLoopStore;
  memref::LoadOp postLoopLoad;
  scf::ForOp enclosingFor;

  for (Operation *user : allocaOp.getResult().getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      auto parentFor = store->getParentOfType<scf::ForOp>();
      if (parentFor && enclosingFor && parentFor != enclosingFor)
        return std::nullopt; // Multiple loops -- bail.

      if (parentFor) {
        enclosingFor = parentFor;
        if (inLoopStore)
          return std::nullopt; // Multiple in-loop stores -- bail.
        inLoopStore = store;
        continue;
      }
      if (preLoopStore)
        return std::nullopt; // Multiple pre-loop stores -- bail.
      preLoopStore = store;
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      auto parentFor = load->getParentOfType<scf::ForOp>();
      if (parentFor && enclosingFor && parentFor != enclosingFor)
        return std::nullopt; // Multiple loops -- bail.

      if (parentFor) {
        enclosingFor = parentFor;
        if (inLoopLoad)
          return std::nullopt; // Multiple in-loop loads -- bail.
        inLoopLoad = load;
        continue;
      }
      if (postLoopLoad)
        return std::nullopt; // Multiple post-loop loads -- bail.
      postLoopLoad = load;
      continue;
    }

    // Unknown user -- bail.
    return std::nullopt;
  }

  // All four components and the enclosing loop must be present.
  if (!preLoopStore || !inLoopLoad || !inLoopStore || !postLoopLoad ||
      !enclosingFor)
    return std::nullopt;

  // Verify positional relationships.
  if (!isBeforeLoop(preLoopStore, enclosingFor))
    return std::nullopt;
  if (!isInsideLoop(inLoopLoad, enclosingFor))
    return std::nullopt;
  if (!isInsideLoop(inLoopStore, enclosingFor))
    return std::nullopt;
  if (!isAfterLoop(postLoopLoad, enclosingFor))
    return std::nullopt;

  // Verify the in-loop load comes before the in-loop store.
  if (!inLoopLoad->isBeforeInBlock(inLoopStore))
    return std::nullopt;

  return LoopCarriedAlloca{allocaOp,    preLoopStore, inLoopLoad,
                           inLoopStore, postLoopLoad, enclosingFor};
}

/// Transform a single scf.for by adding iter_args for qualified allocas.
/// Uses scf::ForOp::replaceWithAdditionalYields to handle the loop body
/// transfer and block argument remapping.
static LogicalResult
promoteAllocas(scf::ForOp forOp, SmallVectorImpl<LoopCarriedAlloca> &allocas) {
  IRRewriter rewriter(forOp.getContext());
  int64_t numExisting = forOp.getNumRegionIterArgs();

  // Collect init values for the new iter_args.
  SmallVector<Value> newInits;
  for (auto &lca : allocas)
    newInits.push_back(lca.preLoopStore.getValueToStore());

  // The callback returns the values to yield for each new iter_arg.
  // These are the values being stored in each iteration.
  NewYieldValuesFn yieldFn = [&](OpBuilder &, Location,
                                 ArrayRef<BlockArgument>) {
    SmallVector<Value> yields;
    for (auto &lca : allocas)
      yields.push_back(lca.inLoopStore.getValueToStore());
    return yields;
  };

  // Replace the loop. This moves the body (not clone), remaps block args,
  // appends yield values, and erases the old loop.
  auto result = forOp.replaceWithAdditionalYields(
      rewriter, newInits, /*replaceInitOperandUsesInLoop=*/false, yieldFn);
  if (failed(result))
    return failure();
  auto newForOp = cast<scf::ForOp>(*result);

  // Replace in-loop loads with the new block args, then erase loads/stores.
  for (auto [idx, lca] : llvm::enumerate(allocas)) {
    Value newBlockArg = newForOp.getRegionIterArgs()[numExisting + idx];
    rewriter.replaceAllUsesWith(lca.inLoopLoad.getResult(), newBlockArg);
    rewriter.eraseOp(lca.inLoopLoad);
    rewriter.eraseOp(lca.inLoopStore);
  }

  // Replace post-loop loads with new loop results.
  for (auto [idx, lca] : llvm::enumerate(allocas)) {
    rewriter.replaceAllUsesWith(lca.postLoopLoad.getResult(),
                                newForOp.getResult(numExisting + idx));
    rewriter.eraseOp(lca.postLoopLoad);
  }

  // Erase pre-loop stores and allocas.
  for (auto &lca : allocas)
    rewriter.eraseOp(lca.preLoopStore);
  for (auto &lca : allocas)
    rewriter.eraseOp(lca.allocaOp);

  return success();
}

//===----------------------------------------------------------------------===//
// PromoteLoopCarriedMemrefs pass
//===----------------------------------------------------------------------===//

struct PromoteLoopCarriedMemrefs
    : public mlir::aster::impl::PromoteLoopCarriedMemrefsBase<
          PromoteLoopCarriedMemrefs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void PromoteLoopCarriedMemrefs::runOnOperation() {
  Operation *rootOp = getOperation();

  // Group qualifying allocas by their enclosing scf.for.
  DenseMap<scf::ForOp, SmallVector<LoopCarriedAlloca>> loopToAllocas;

  rootOp->walk([&](memref::AllocaOp allocaOp) {
    if (auto lca = classifyAlloca(allocaOp))
      loopToAllocas[lca->forOp].push_back(*lca);
  });

  if (loopToAllocas.empty())
    return;

  // Collect loops. In practice there is typically one qualifying loop (the
  // K-loop), but we handle multiple by processing inner-to-outer.
  SmallVector<scf::ForOp> loops;
  for (auto &[forOp, _] : loopToAllocas)
    loops.push_back(forOp);
  // Sort inner-before-outer: a proper ancestor should come after its
  // descendants. For sibling loops in the same block, process later ones
  // first so erasing doesn't affect earlier positions.
  llvm::sort(loops, [](scf::ForOp a, scf::ForOp b) {
    if (b->isProperAncestor(a))
      return true; // a is inside b -> process a first
    if (a->isProperAncestor(b))
      return false; // b is inside a -> process b first
    // Same block: process later ops first (reverse source order).
    if (a->getBlock() == b->getBlock())
      return b->isBeforeInBlock(a);
    return false; // unrelated blocks -- order doesn't matter
  });

  for (scf::ForOp forOp : loops) {
    auto &allocas = loopToAllocas[forOp];
    if (failed(promoteAllocas(forOp, allocas))) {
      forOp.emitWarning("failed to promote loop-carried memrefs");
      return signalPassFailure();
    }
  }
}
