//===- CanonicalizeWaits.cpp - Canonicalize wait operations --------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstdint>
#include <utility>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNCANONICALIZEWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AMDGCNCanonicalizeWaits pass
//===----------------------------------------------------------------------===//
struct AMDGCNCanonicalizeWaits
    : public mlir::aster::amdgcn::impl::AMDGCNCanonicalizeWaitsBase<
          AMDGCNCanonicalizeWaits> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// WaitCanonicalizer
//===----------------------------------------------------------------------===//

struct WaitCanonicalizer {
  WaitCanonicalizer(bool useLICM) : useLICM(useLICM) {}
  LogicalResult runTransforms(Operation *op, DominanceInfo *domInfo);

private:
  /// Fission wait op dependencies by loop ancestors. Creates new wait ops for
  /// dependencies defined outside each loop level.
  void fissionWaitOpByLoopAncestors(amdgcn::WaitOp waitOp, OpBuilder &builder);

  /// Canonicalize wait ops after transformations.
  void canonicalizeWaits(DataFlowSolver &solver);
  /// Whether to use LICM to hoist wait ops out of loops.
  bool useLICM;

  /// Temporary storage.
  SmallVector<Value, 5> scratch1, scratch2;
  /// Collected wait ops.
  SmallVector<WaitOp> waitOps;
};
} // namespace

void WaitCanonicalizer::fissionWaitOpByLoopAncestors(amdgcn::WaitOp waitOp,
                                                     OpBuilder &builder) {
  // Collect loop ancestors.
  SmallVector<LoopLikeOpInterface> loopAncestors;
  auto loopLike = waitOp->getParentOfType<LoopLikeOpInterface>();
  while (loopLike) {
    loopAncestors.push_back(loopLike);
    loopLike = loopLike->getParentOfType<LoopLikeOpInterface>();
  }

  // If there are no loop ancestors, nothing to do.
  if (loopAncestors.empty())
    return;

  // Split operands by loop in reverse order.
  SmallVector<std::pair<SmallVector<Value, 5>, LoopLikeOpInterface>> argGroups;
  SmallVectorImpl<Value> &scratch = scratch1, &argList = scratch2;
  argList.assign(waitOp.getDependencies().begin(),
                 waitOp.getDependencies().end());
  for (LoopLikeOpInterface loopLike : llvm::reverse(loopAncestors)) {
    scratch.clear();
    for (Value &arg : argList) {
      if (!arg)
        continue;

      Operation *aOp = arg.getDefiningOp();
      if (!aOp) {
        aOp = cast<BlockArgument>(arg).getOwner()->getParentOp();
      }
      assert(aOp && "expected valid defining operation for wait dependency");
      if (loopLike->isAncestor(aOp))
        continue;
      scratch.push_back(arg);
      arg = Value();
    }
    if (!scratch.empty())
      argGroups.push_back({std::move(scratch), loopLike});
  }

  // Bail out if no fissioning is needed.
  if (argGroups.size() == 0)
    return;

  // Remove null entries from argList.
  argList.erase(std::remove(argList.begin(), argList.end(), Value()),
                argList.end());

  // Create new wait ops for each group.
  builder.setInsertionPoint(waitOp);
  for (const auto &[argGroup, loopLike] : argGroups) {
    if (!useLICM)
      builder.setInsertionPoint(loopLike);
    amdgcn::WaitOp::create(builder, waitOp.getLoc(), argGroup);
  }
  if (argList.empty() && !waitOp.hasAnyCount()) {
    waitOp.erase();
    return;
  }
  waitOp.getDependenciesMutable().assign(argList);
}

void WaitCanonicalizer::canonicalizeWaits(DataFlowSolver &solver) {
  for (WaitOp op : waitOps) {
    const auto *beforeState =
        solver.lookupState<WaitState>(solver.getProgramPointBefore(op));
    assert(beforeState &&
           "expected valid wait analysis states before and after wait op");
    // Remove wait op if all dependencies are already waited on.
    if (beforeState->isEmpty()) {
      op->erase();
      continue;
    }
    const auto *afterState =
        solver.lookupState<WaitState>(solver.getProgramPointAfter(op));
    assert(afterState &&
           "expected valid wait analysis states before and after wait op");
    const std::optional<WaitOpInfo> &waitInfo = afterState->getWaitOpInfo();

    if (!waitInfo.has_value()) {
      if (op.isNop())
        op->erase();
      return;
    }

    std::array<TokenState::Position, 3> minPos = {
        op.getVmCnt(),   // VMEM
        op.getLgkmCnt(), // SMEM
        op.getLgkmCnt(), // DMEM
    };
    SmallVector<Value> newToks;
    for (const TokenState &tok : waitInfo->getWaitedTokens()) {
      if (Value v = tok.getToken()) {
        newToks.push_back(v);
        continue;
      }
      assert(tok.getID() == TokenState::kUnknownID &&
             "expected unknown ID for null token");
      int32_t i = static_cast<int32_t>(tok.getKind()) - 1;
      assert(i >= 0 && i < 3 && "invalid memory kind");
      // minPos[i] = std::min(minPos[i], tok.getPosition());
    }
    op.getDependenciesMutable().assign(newToks);
    if (op.isNop())
      op->erase();
  }
}

LogicalResult WaitCanonicalizer::runTransforms(Operation *op,
                                               DominanceInfo *domInfo) {
  OpBuilder builder(op->getContext());

  op->walk([&](amdgcn::WaitOp waitOp) {
    // Fission operands by whether they are defined outside of a loop.
    fissionWaitOpByLoopAncestors(waitOp, builder);
  });

  // Optionally run LICM to move wait ops out of loops.
  if (useLICM) {
    op->walk([&](LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          [&](Value value, Region *) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *) {
            if (isPure(op))
              return true;
            auto wOp = dyn_cast<amdgcn::WaitOp>(op);
            return wOp && (!wOp.hasAnyCount());
          },
          [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
    });
  }

  if (!domInfo)
    return success();

  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<WaitAnalysis>(*domInfo);

  // Initialize and run the solver on the kernel
  if (failed(solver.initializeAndRun(op)))
    return op->emitError() << "failed to run wait analysis";

  // Remove already-waited dependencies.
  canonicalizeWaits(solver);
  return success();
}

//===----------------------------------------------------------------------===//
// AMDGCNCanonicalizeWaits pass
//===----------------------------------------------------------------------===//

LogicalResult amdgcn::canonicalizeWaits(Operation *op, DominanceInfo *domInfo,
                                        bool useLICM) {
  return WaitCanonicalizer(useLICM).runTransforms(op, domInfo);
}

void AMDGCNCanonicalizeWaits::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();
  if (failed(canonicalizeWaits(getOperation(), &domInfo, useLICM)))
    return signalPassFailure();
}
