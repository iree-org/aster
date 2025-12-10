//===- AMDGCNOptimizeStraightLineWaits.cpp - Optimize Straight Line Waits -===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-optimize-straight-line-waits"

namespace mlir {
namespace aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNOPTIMIZESTRAIGHTLINEWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::lsir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if an operation is a branch
static bool isBranchOp(Operation &op) {
  auto branchOp = dyn_cast<BranchOpInterface>(&op);
  return branchOp != nullptr && branchOp->getNumSuccessors() > 0;
}

/// Check if a block has no branch operations
static bool hasNoBranches(Block *block) {
  if (!block)
    return false;
  for (Operation &op : *block) {
    if (isBranchOp(op))
      return false;
  }
  return true;
}

/// Erase all waitcnt operations in the block
static void eraseWaitcntOpsInBlock(Block *block) {
  llvm::SmallVector<Operation *> waitcntOpsToErase;
  block->walk([&](Operation *op) {
    if (auto wOp = dyn_cast<amdgcn::inst::SWaitcntOp>(op);
        wOp && !wOp.getImmutable())
      waitcntOpsToErase.push_back(op);
  });
  for (Operation *op : waitcntOpsToErase)
    op->erase();
}

//===----------------------------------------------------------------------===//
// AMDGCNOptimizeStraightLineWaits pass
//===----------------------------------------------------------------------===//
namespace {
struct AMDGCNOptimizeStraightLineWaits
    : public amdgcn::impl::AMDGCNOptimizeStraightLineWaitsBase<
          AMDGCNOptimizeStraightLineWaits> {
public:
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNOptimizeStraightLineWaits pass implementation
//===----------------------------------------------------------------------===//
void AMDGCNOptimizeStraightLineWaits::runOnOperation() {
  Operation *op = getOperation();

  // if the op has more than one block or if the block is either in a loop or
  // has branches just skip
  if (op->getNumRegions() != 1 || op->getRegion(0).getBlocks().size() != 1)
    return;

  Block *block = &op->getRegion(0).getBlocks().front();
  if (!hasNoBranches(block) || LoopLikeOpInterface::blockIsInLoop(block))
    return;

  // Erase all waitcnt ops in the block, we're taking ownership here.
  eraseWaitcntOpsInBlock(block);

  // Create and configure the data flow solver for this block.
  LDBG() << "Processing op: " << *op;
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<MemoryDependenceAnalysis>(flushAllMemoryOnExit);
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "Failed to run data flow analysis";
    return signalPassFailure();
  }

  // Process the single top-level block (already validated there's only 1).
  // Query the program point after each operation and insert a waitcnt if
  // there are any must flush before op memory operations.
  MLIRContext *ctx = op->getContext();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(block);
  block->walk([&](Operation *op) {
    auto *beforeState = solver.lookupState<MemoryDependenceLattice>(
        solver.getProgramPointBefore(op));
    auto *afterState = solver.lookupState<MemoryDependenceLattice>(
        solver.getProgramPointAfter(op));
    if (!afterState || afterState->getMustFlushBeforeOpCount() == 0) {
      LDBG() << "No waitcnt needed for " << *op;
      return;
    }

    // Count the number of pending flush ops on Global and LDS resources.
    // This is tracked by the before state (because the after state contains the
    // new op).
    int pendingFlushLgkmcnt = 0, pendingFlushVmcnt = 0, pendingFlushExpcnt = 0;
    for (const auto &memOp : beforeState->getPendingAfterOp()) {
      if (memOp.resourceType == GlobalMemoryResource::get())
        pendingFlushVmcnt++;
      if (memOp.resourceType == LDSMemoryResource::get())
        pendingFlushLgkmcnt++;
    }

    // Count the number of must flush ops on Global and LDS resources.
    // This is tracked by the after state (because dataflow is forward and the
    // beforeState is const by the time we have info to update).
    int mustFlushLgkmcnt = 0, mustFlushVmcnt = 0, mustFlushExpcnt = 0;
    for (const auto &memOp : afterState->getMustFlushBeforeOp()) {
      if (memOp.resourceType == GlobalMemoryResource::get())
        mustFlushVmcnt++;
      if (memOp.resourceType == LDSMemoryResource::get())
        mustFlushLgkmcnt++;
    }

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    int vmcnt = pendingFlushVmcnt - mustFlushVmcnt;
    int expcnt = pendingFlushExpcnt - mustFlushExpcnt;
    int lgkmcnt = pendingFlushLgkmcnt - mustFlushLgkmcnt;
    Operation *sWaitcntOp = S_WAITCNT::create(b, op->getLoc(), vmcnt, expcnt,
                                              std::min(lgkmcnt, 15));
    LDBG() << "Inserted waitcnt: " << *sWaitcntOp;
  });

  // At the end of the pass, replace lsir.assume_noalias results with its
  // operands: just forward the operands This serves as a poor man's
  // "late lowering" of lsir.assume_noalias ops.
  SmallVector<Operation *> assumeNoaliasOpsToErase;
  block->walk([&](Operation *op) {
    if (isa<lsir::AssumeNoaliasOp>(op)) {
      for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
        op->getResult(idx).replaceAllUsesWith(operand);
      }
      assumeNoaliasOpsToErase.push_back(op);
    }
  });
  for (Operation *op : assumeNoaliasOpsToErase)
    op->erase();
}
