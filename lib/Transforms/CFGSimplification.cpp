//===- CFGSimplification.cpp - CFG Simplification -------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements two CFG simplification transformations:
//
// 1. Block merging: if BBSucc has BBPred as its only predecessor and BBPred
//    has BBSucc as its only successor (determined via BranchOpInterface),
//    BBSucc is merged into BBPred.
//
// 2. Empty block elimination: if a block contains only a single-successor
//    terminator, all predecessors are redirected to branch directly to the
//    successor and the now-unreachable block is removed.
//
// Both transformations are applied to fixpoint within each region.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
#define GEN_PASS_DEF_CFGSIMPLIFICATION
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct CFGSimplification
    : public aster::impl::CFGSimplificationBase<CFGSimplification> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Try to merge each block into its unique predecessor. A block BBSucc is
/// merged into BBPred when:
/// - BBSucc has exactly one predecessor, BBPred.
/// - BBPred's terminator is a BranchOpInterface with exactly one successor.
/// Returns true if at least one merge was performed.
static bool mergeIntoUniquePredecessor(Region &region, IRRewriter &rewriter) {
  SmallVector<Block *> candidates;
  for (Block &block : region) {
    if (!block.isEntryBlock())
      candidates.push_back(&block);
  }

  bool changed = false;
  for (Block *block : candidates) {
    Block *pred = block->getSinglePredecessor();
    // Skip if there is no predecessor.
    if (!pred)
      continue;

    Operation *term = pred->getTerminator();
    auto brOp = dyn_cast<BranchOpInterface>(term);

    // Skip if the terminator is not a branch or has multiple successors.
    if (!brOp || term->getNumSuccessors() != 1)
      continue;

    assert(term->getSuccessor(0) == block &&
           "single successor must be this block");

    // Get the successor operands.
    SuccessorOperands operands = brOp.getSuccessorOperands(0);

    // Skip if the terminator produces any operands.
    if (operands.getProducedOperandCount() > 0)
      continue;

    // Merge the block into the predecessor.
    SmallVector<Value> forwardedArgs =
        llvm::to_vector(operands.getForwardedOperands());
    rewriter.eraseOp(term);
    rewriter.mergeBlocks(block, pred, forwardedArgs);
    changed = true;
  }
  return changed;
}

/// Try to eliminate empty blocks (those containing only a single-successor
/// terminator). Every predecessor is redirected to branch directly to the
/// successor, substituting any forwarded block-argument values. The empty
/// block is then erased.
/// Returns true if at least one block was eliminated.
static bool eliminateEmptyBlocks(Region &region, IRRewriter &rewriter) {
  SmallVector<Block *> blocksToErase;

  // A list with the block to forward to, the forwarded operands, the branch to
  // update, and the successor index.
  SmallVector<std::tuple<Block *, ValueRange, BranchOpInterface, int32_t>>
      updateWorklist;

  // Collect all blocks to update.
  for (Block &block : region) {
    // Skip entry blocks and empty blocks.
    if (block.isEntryBlock() || block.empty())
      continue;

    // Skip if the block contains any other operations.
    if (&block.front() != &block.back())
      continue;
    auto brOp = dyn_cast<BranchOpInterface>(block.getTerminator());

    // Skip if the terminator is not a branch, has multiple successors, or is a
    // self-loop.
    if (!brOp || brOp->getNumSuccessors() != 1 ||
        brOp->getSuccessor(0) == &block)
      continue;

    // Get the successor information.
    SuccessorOperands successorOperands = brOp.getSuccessorOperands(0);
    ValueRange brOperands = successorOperands.getForwardedOperands();
    Block *successor = brOp->getSuccessor(0);

    // Skip if the terminator produces any operands.
    if (successorOperands.getProducedOperandCount() > 0)
      continue;

    bool skipBlock = false;
    int64_t pos = updateWorklist.size();

    // Collect all blocks predecessors to update.
    for (Block *pred : block.getPredecessors()) {
      auto predBrOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
      // Skip if the predecessor does not have a branch terminator.
      if (!predBrOp || skipBlock) {
        skipBlock = true;
        break;
      }

      // Collect all branches to update.
      for (int64_t idx = 0, e = predBrOp->getNumSuccessors(); idx < e; ++idx) {
        // Skip if the successor is not the current block.
        if (predBrOp->getSuccessor(idx) != &block)
          continue;

        SuccessorOperands successorOperands =
            predBrOp.getSuccessorOperands(idx);

        // Skip if the successor produces any operands.
        if (successorOperands.getProducedOperandCount() > 0) {
          skipBlock = true;
          break;
        }

        // Add the branch to the update worklist.
        updateWorklist.push_back({successor, brOperands, predBrOp, idx});
      }
    }

    // Skip if the block is not eligible for elimination.
    if (skipBlock) {
      // Remove all inserted entries for this block.
      updateWorklist.erase(updateWorklist.begin() + pos, updateWorklist.end());
      continue;
    }

    blocksToErase.push_back(&block);
  }

  if (blocksToErase.empty())
    return false;

  // Process each block candidate for elimination.
  llvm::SmallDenseMap<Value, Value> valueMap;
  for (auto [succ, fwdOperands, brOp, idx] : updateWorklist) {
    valueMap.clear();

    // Map block arguments to the predecessor operands.
    Block *block = brOp->getSuccessor(idx);
    SuccessorOperands successorOperands = brOp.getSuccessorOperands(idx);
    for (auto [bbArg, predVal] : llvm::zip_equal(
             block->getArguments(), successorOperands.getForwardedOperands()))
      valueMap.insert({bbArg, predVal});

    // Build the new forwarded operands.
    SmallVector<Value> newOperands;
    newOperands.reserve(fwdOperands.size());
    for (Value operand : fwdOperands)
      newOperands.push_back(valueMap.lookup_or(operand, operand));

    // Update the branch.
    successorOperands.getMutableForwardedOperands().assign(newOperands);
    brOp->setSuccessor(succ, idx);
  }

  // Erase the blocks.
  for (Block *block : blocksToErase)
    rewriter.eraseBlock(block);

  return true;
}

void CFGSimplification::runOnOperation() {
  IRRewriter rewriter(&getContext());

  // Update the CFG until fixpoint.
  getOperation()->walk<WalkOrder::PostOrder>([&](Region *region) {
    if (region->empty())
      return;
    bool changed = true;
    while (changed) {
      changed = false;
      changed |= mergeIntoUniquePredecessor(*region, rewriter);
      changed |= eliminateEmptyBlocks(*region, rewriter);
    }
  });
}
