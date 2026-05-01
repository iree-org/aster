// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This pass hoists amdgcn.wait ops whose token dependencies come from
// scf.for iter_args to the top of the loop body. This opens up the
// scheduling window for the low-level instruction scheduler by removing
// artificial wait barriers from the middle of the loop body.
//
// For waits with mixed dependencies (some iter_arg, some intra-iteration),
// the wait is cloned: a new wait at the loop head takes the iter_arg deps,
// and the original wait retains only the intra-iteration deps.
//
// After this pass, a subsequent canonicalize merges adjacent hoisted waits.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-hoist-iter-arg-waits"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_HOISTITERARGWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

struct HoistIterArgWaits
    : public amdgcn::impl::HoistIterArgWaitsBase<HoistIterArgWaits> {
  using Base::Base;
  void runOnOperation() override;
};

/// Check if a block argument is a token-type iter_arg of the given scf.for.
static bool isTokenIterArg(BlockArgument arg, scf::ForOp forOp) {
  if (arg.getOwner() != forOp.getBody())
    return false;
  if (arg == forOp.getInductionVar())
    return false;
  return isa<TokenDependencyTypeInterface>(arg.getType());
}

/// Process a single scf.for: hoist iter_arg waits to the loop head.
static bool processForOp(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  bool changed = false;

  // insertionPoint tracks the end of the already-hoisted prefix.
  Operation *insertionPoint = &body->front();

  // Direct children only (no recursion into nested loops).
  SmallVector<WaitOp> waitsToProcess(body->getOps<WaitOp>());

  for (WaitOp wait : waitsToProcess) {
    // Already in the hoisted prefix; advance past it.
    if (wait.getOperation() == insertionPoint) {
      insertionPoint = wait->getNextNode();
      continue;
    }

    // Classify dependencies as iter_arg or intra-iteration.
    SmallVector<Value> iterArgDeps;
    SmallVector<Value> intraDeps;
    for (Value dep : wait.getDependencies()) {
      auto blockArg = dyn_cast<BlockArgument>(dep);
      if (blockArg && isTokenIterArg(blockArg, forOp))
        iterArgDeps.push_back(dep);
      else
        intraDeps.push_back(dep);
    }

    // No iter_arg deps: skip this wait entirely.
    if (iterArgDeps.empty())
      continue;

    if (intraDeps.empty()) {
      LDBG() << "Hoisting pure iter_arg wait: " << *wait;
      wait->moveBefore(insertionPoint);
      insertionPoint = wait->getNextNode();
      changed = true;
      continue;
    }

    // Mixed deps: clone with iter_arg deps at the top, original keeps
    // only intra-iteration deps.
    LDBG() << "Cloning mixed-dep wait: " << *wait;
    OpBuilder builder(insertionPoint);
    auto hoistedWait = builder.create<WaitOp>(
        wait.getLoc(), iterArgDeps, WaitOp::kNoWaitCount, WaitOp::kNoWaitCount);
    LDBG() << "  Hoisted: " << *hoistedWait;
    wait.setDependencies(intraDeps);
    LDBG() << "  Original kept: " << *wait;
    insertionPoint = hoistedWait->getNextNode();
    changed = true;
  }

  // Step 2: Move barriers that have no waits after them in the loop body.
  // After hoisting, if all waits that preceded a barrier were moved to the
  // top, the barrier has no waits after it and can follow the hoisted waits.
  for (auto &op : llvm::make_early_inc_range(*body)) {
    auto barrierOp = dyn_cast<SBarrier>(&op);
    if (!barrierOp)
      continue;
    // Already in the hoisted prefix.
    if (barrierOp->isBeforeInBlock(insertionPoint) ||
        barrierOp.getOperation() == insertionPoint)
      continue;
    // Check: are there any WaitOps after this barrier?
    bool hasWaitAfter = false;
    for (auto it = std::next(Block::iterator(barrierOp)), end = body->end();
         it != end; ++it) {
      if (isa<WaitOp>(&*it)) {
        hasWaitAfter = true;
        break;
      }
    }
    if (!hasWaitAfter) {
      LDBG() << "Moving barrier after hoisted waits: " << *barrierOp;
      barrierOp->moveBefore(insertionPoint);
      insertionPoint = barrierOp->getNextNode();
      changed = true;
    }
  }

  return changed;
}

void HoistIterArgWaits::runOnOperation() {
  bool changed = false;
  getOperation()->walk([&](scf::ForOp forOp) {
    if (llvm::any_of(forOp.getRegionIterArgs(), [](BlockArgument arg) {
          return isa<TokenDependencyTypeInterface>(arg.getType());
        }))
      changed |= processForOp(forOp);
  });

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace
