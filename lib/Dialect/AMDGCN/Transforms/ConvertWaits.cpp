//===- ConvertWaits.cpp - Convert wait ops to hardware instructions ------===//
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
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNCONVERTWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct AMDGCNConvertWaits
    : public mlir::aster::amdgcn::impl::AMDGCNConvertWaitsBase<
          AMDGCNConvertWaits> {
public:
  using Base::Base;
  void runOnOperation() override;

  /// Remove token arguments from control-flow arguments.
  LogicalResult removeTokenArguments(FunctionOpInterface funcOp);
};
} // namespace

void AMDGCNConvertWaits::runOnOperation() {
  Operation *op = getOperation();

  // Get dominance info for the analysis.
  auto &domInfo = getAnalysis<DominanceInfo>();

  // Create and configure the data flow solver.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<WaitAnalysis>(domInfo);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "failed to run wait analysis";
    return signalPassFailure();
  }

  // Rewrite wait operations based on analysis results.
  IRRewriter rewriter(op->getContext());
  op->walk([&](WaitOp waitOp) {
    const auto *afterState =
        solver.lookupState<WaitState>(solver.getProgramPointAfter(waitOp));
    assert(afterState &&
           "expected valid wait analysis states before and after wait op");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(waitOp);
    WaitCnt counts = afterState->waitOpInfo->counts;

    // TODO: support hasExpcnt.
    bool hasExpcnt = false;
    bool hasVmcnt = counts.getVmcnt() < TokenState::kMaxPosition;
    bool hasLgkmcnt = counts.getLgkmcnt() < TokenState::kMaxPosition;
    // If no wait is needed, erase the wait op.
    if (!hasVmcnt && !hasLgkmcnt && !hasExpcnt) {
      rewriter.eraseOp(waitOp);
      return;
    }

    // Replace the wait op with an inst::SWaitcntOp with the minimal counts
    // required after this wait op.
    // Note that `SWaitcntOp` defaults to max counts, so we need to:
    //   1. set the counts that are less than max and hasXXX is true
    //   2. leave it alone when hasXXX is true
    //   3. unset the attribute when hasXXX is false
    auto newWait = rewriter.replaceOpWithNewOp<inst::SWaitcntOp>(waitOp);

    // The default builder of SWaitcntOp sets both counts to max, so we take the
    // minimum of the max allowed hardware value count and the required count.
    if (hasVmcnt) {
      newWait.setVmcnt(std::min(static_cast<int32_t>(newWait.getVmcnt()) - 1,
                                counts.getVmcnt()));
    }
    if (hasLgkmcnt) {
      newWait.setLgkmcnt(std::min(
          static_cast<int32_t>(newWait.getLgkmcnt()) - 1, counts.getLgkmcnt()));
    }
    if (hasExpcnt) {
      llvm_unreachable("EXPCNT is not supported yet");
    }
  });

  // Check if we should remove token arguments from control-flow arguments.
  if (!removeTokenArgs)
    return;

  op->walk([&](FunctionOpInterface funcOp) {
    if (failed(removeTokenArguments(funcOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

LogicalResult
AMDGCNConvertWaits::removeTokenArguments(FunctionOpInterface funcOp) {
  IRRewriter rewriter(funcOp->getContext());
  Region &region = funcOp.getFunctionBody();

  // Create a helper to get a poison value for a given type, and insert it at
  // the start of the entry block.
  Block *block = &region.front();
  DenseMap<Type, Value> poisonCache;
  Location loc = funcOp.getLoc();
  auto getPoison = [&](Type type) -> Value {
    Value &poison = poisonCache[type];
    if (poison)
      return poison;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    poison = ub::PoisonOp::create(rewriter, loc, type);
    return poison;
  };

  // Replace token operands with poison values which carry MLIR semantics for
  // "this value is intentionally meaningless".
  // This is a much simpler alternative to implement than removing all uses,
  // block args, branch operands, and op signatures in one coordinated pass.
  auto poisonTokenOperands = [&](Operation *op) {
    for (OpOperand &use : op->getOpOperands()) {
      if (!isa<ReadTokenType, WriteTokenType>(use.get().getType()))
        continue;
      use.set(getPoison(use.get().getType()));
    }
  };

  // Check if an argument should be removed.
  auto shouldRemoveArg = +[](Value arg) {
    return isa<ReadTokenType, WriteTokenType>(arg.getType()) && arg.use_empty();
  };

  SmallVector<Operation *> opsToCanonicalize;
  // Walk all blocks and handle token arguments, region branch operands, and
  // terminator operands.
  funcOp->walk([&](Block *block) {
    if (block->empty())
      return;

    bool changed = false;
    // Handle block arguments.
    for (Value arg : block->getArguments()) {
      if (!isa<ReadTokenType, WriteTokenType>(arg.getType()))
        continue;
      rewriter.replaceAllUsesWith(arg, getPoison(arg.getType()));
      changed = true;
    }

    // Erase token block arguments if not in the entry block.
    if (changed && !block->isEntryBlock())
      block->eraseArguments(shouldRemoveArg);

    // Handle region branch operands.
    for (auto brOp : block->getOps<RegionBranchOpInterface>()) {
      opsToCanonicalize.push_back(brOp);
      poisonTokenOperands(brOp);
    }

    // Handle branch operands.
    if (auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator())) {
      for (auto [idx, succ] : llvm::enumerate(brOp->getSuccessors())) {
        SuccessorOperands succOperands = brOp.getSuccessorOperands(idx);
        assert(succOperands.getProducedOperandCount() == 0 &&
               "expected no produced operands");
        MutableOperandRange forwarded =
            succOperands.getMutableForwardedOperands();
        for (int64_t i = static_cast<int64_t>(forwarded.size()) - 1; i >= 0;
             --i) {
          if (isa<ReadTokenType, WriteTokenType>(forwarded[i].get().getType()))
            forwarded.erase(i);
        }
      }
      return;
    }

    // Skip function returns.
    if (isa<FunctionOpInterface>(block->getParentOp()))
      return;

    // Handle all other terminator operands.
    poisonTokenOperands(block->getTerminator());
  });

  // Collect all canonicalization patterns for region branch ops.
  RewritePatternSet patterns(funcOp->getContext());
  DenseSet<RegisteredOperationName> populatedPatterns;
  for (Operation *op : opsToCanonicalize) {
    if (std::optional<RegisteredOperationName> info = op->getRegisteredInfo())
      if (populatedPatterns.insert(*info).second)
        info->getCanonicalizationPatterns(patterns, op->getContext());
  }

  // Canonicalize all region branch ops.
  if (failed(applyOpPatternsGreedily(opsToCanonicalize, std::move(patterns)))) {
    funcOp->emitError("greedy pattern rewrite failed to converge");
    signalPassFailure();
    return failure();
  }
  return success();
}
