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
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/DependentOpInterface.h"
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

/// Rewrite a gfx1250 wait op to per-counter SOPP instructions. Load+ds (or
/// store+ds) fuse into the combined s_wait_{load,store}cnt_dscnt form; loadcnt
/// takes priority for the ds pairing; km and tensor never fuse.
static void rewriteGfx1250Wait(IRRewriter &rewriter, Operation *waitOpOp,
                               const WaitCnt &counts,
                               SmallVectorImpl<Operation *> &toErase) {
  Location loc = waitOpOp->getLoc();
  const WaitCntGfx1250 &c = std::get<WaitCntGfx1250>(counts);
  auto get = [&](WaitCounterKind s) { return c.getCount(s); };
  auto present = [](int32_t c) { return c < TokenState::kMaxPosition; };
  int32_t load = get(WaitCounterKind::Load);
  int32_t store = get(WaitCounterKind::Store);
  int32_t ds = get(WaitCounterKind::Ds);
  bool loadDone = false, storeDone = false, dsDone = false;
  if (present(load) && present(ds)) {
    SWaitLoadcntDscnt::create(rewriter, loc).setCount((load << 8) | ds);
    loadDone = dsDone = true;
  } else if (present(store) && present(ds)) {
    SWaitStorecntDscnt::create(rewriter, loc).setCount((store << 8) | ds);
    storeDone = dsDone = true;
  }
  if (present(load) && !loadDone)
    SWaitLoadcnt::create(rewriter, loc).setCount(load);
  if (present(store) && !storeDone)
    SWaitStorecnt::create(rewriter, loc).setCount(store);
  if (present(ds) && !dsDone)
    SWaitDscnt::create(rewriter, loc).setCount(ds);
  if (present(get(WaitCounterKind::ScalarRead)))
    SWaitKmcnt::create(rewriter, loc)
        .setCount(get(WaitCounterKind::ScalarRead));
  if (present(get(WaitCounterKind::Tensor)))
    SWaitTensorcnt::create(rewriter, loc)
        .setCount(get(WaitCounterKind::Tensor));
  toErase.push_back(waitOpOp);
}

/// Rewrite a CDNA wait op to a single s_waitcnt with merged vm/lgkm counts.
static void rewriteCdnaWait(IRRewriter &rewriter, Operation *waitOpOp,
                            const WaitCnt &counts,
                            SmallVectorImpl<Operation *> &toErase) {
  auto present = [](int32_t c) { return c < TokenState::kMaxPosition; };
  const WaitCntCdna3 &c = std::get<WaitCntCdna3>(counts);
  int32_t vm = c.vmcnt;
  int32_t lgkm = c.lgkmcnt;
  bool hasVm = present(vm), hasLgkm = present(lgkm);
  // Create the s_waitcnt without optional fence_token then mark the wait for
  // erasure.
  if (!hasVm && !hasLgkm) {
    toErase.push_back(waitOpOp);
    return;
  }
  auto newWait = SWaitcnt::create(rewriter, waitOpOp->getLoc());
  if (hasVm)
    newWait.setVmcnt(
        std::min(static_cast<int32_t>(newWait.getVmcnt()) - 1, vm));
  if (hasLgkm)
    newWait.setLgkmcnt(
        std::min(static_cast<int32_t>(newWait.getLgkmcnt()) - 1, lgkm));
  toErase.push_back(waitOpOp);
}

/// Run the wait analysis on `root` with `model`, then rewrite every wait op
/// inside it to its hardware form.
static LogicalResult convertWaitsOn(Operation *root, ISAVersion isaVersion,
                                    DominanceInfo &domInfo) {
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  loadWaitAnalysis(solver, domInfo, isaVersion);
  if (failed(solver.initializeAndRun(root))) {
    root->emitError() << "failed to run wait analysis";
    return failure();
  }
  // Note: FenceToken return values disappear during legalization; strip
  // consumer fence_token operands before erasing wait/barrier producers.
  SmallVector<Operation *> toErase;
  IRRewriter rewriter(root->getContext());
  root->walk([&](WaitCntOpInterface waitOp) {
    const auto *afterState =
        solver.lookupState<WaitState>(solver.getProgramPointAfter(waitOp));
    assert(afterState &&
           "expected valid wait analysis states before and after wait op");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(waitOp);
    const WaitCnt &counts = afterState->waitOpInfo->counts;
    if (isa<WaitGfx1250Op>(waitOp))
      rewriteGfx1250Wait(rewriter, waitOp, counts, toErase);
    else
      rewriteCdnaWait(rewriter, waitOp, counts, toErase);
  });
  root->walk([&](CrossWaveTokenBarrierOp barrier) {
    rewriter.setInsertionPoint(barrier);
    if (needsSplitBarriers(isaVersion)) {
      SBarrierSignal::create(rewriter, barrier.getLoc(),
                             static_cast<uint16_t>(-1));
      SBarrierWait::create(rewriter, barrier.getLoc(),
                           static_cast<uint16_t>(-1));
    } else {
      SBarrier::create(rewriter, barrier.getLoc());
    }
    toErase.push_back(barrier);
  });
  root->walk([&](FenceableOpInterface fenceable) {
    if (!fenceable.hasFenceToken())
      return;
    rewriter.modifyOpInPlace(fenceable, [&]() { fenceable.eraseFence(); });
  });
  for (Operation *op : llvm::reverse(toErase))
    rewriter.eraseOp(op);
  return success();
}

void AMDGCNConvertWaits::runOnOperation() {
  amdgcn::ModuleOp module = getOperation();

  // Apply canonicalization patterns to clean up wait operations.
  RewritePatternSet patterns(module.getContext());
  WaitOp::getCanonicalizationPatterns(patterns, module.getContext());
  WaitGfx1250Op::getCanonicalizationPatterns(patterns, module.getContext());
  if (failed(applyPatternsGreedily(
          module, std::move(patterns),
          GreedyRewriteConfig()
              .enableFolding(true)
              .enableConstantCSE(true)
              .setUseTopDownTraversal(true)
              .setRegionSimplificationLevel(
                  GreedySimplifyRegionLevel::Disabled)))) {
    module.emitError() << "Failed to apply wait op canonicalization patterns";
    return signalPassFailure();
  }

  auto &domInfo = getAnalysis<DominanceInfo>();
  if (failed(convertWaitsOn(module, getIsaForOp(module), domInfo)))
    return signalPassFailure();

  if (!removeTokenArgs)
    return;
  module.walk([&](FunctionOpInterface funcOp) {
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
