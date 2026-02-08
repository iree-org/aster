//===- Bufferization.cpp -------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs value provenance analysis and inserts phi-breaking copies
// before branches where multiple allocas merge at block arguments.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-bufferization"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_BUFFERIZATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// Bufferization pass
//===----------------------------------------------------------------------===//
struct Bufferization : public amdgcn::impl::BufferizationBase<Bufferization> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// CFGSimplifier
//===----------------------------------------------------------------------===//
/// Comparator for the control-flow pairs.
struct CmpCFPair {
  bool operator()(const std::pair<BlockArgument, OpOperand *> &lhs,
                  const std::pair<BlockArgument, OpOperand *> &rhs) const {
    return std::make_tuple(lhs.first.getAsOpaquePointer(), lhs.second) <
           std::make_tuple(rhs.first.getAsOpaquePointer(), rhs.second);
  }
};

/// Simplify the control-flow graph by removing block arguments with register
/// value semantics. This is done by:
/// 1. Computing the control-flow pairs between the block arguments and the
/// operands.
/// 2. Reducing the control-flow pairs by removing the irreducible operands and
/// block arguments.
/// 3. Removing the block arguments by inserting phi-breaking copies.
/// 4. Removing the operands by erasing the operands from the branch operands.
/// An irreducible operand is an operand that cannot be safely removed.
struct CFGSimplifier {
  CFGSimplifier(IRRewriter &rewriter, DominanceInfo &domInfo, Block *entryBlock)
      : rewriter(rewriter), domInfo(domInfo), entryBlock(entryBlock) {}
  void run(Region &region);

private:
  /// Compute the control-flow pairs between the block arguments and the
  /// operands of the terminator.
  void computeControlFlowPairs(Block *block,
                               DenseSet<void *> &irreducibleElements,
                               DenseSet<BlockArgument> &visitedBBArgs);
  /// Remove control-flow pairs that cannot be safely removed.
  void reduceControlFlowPairs(DenseSet<void *> &irreducibleElements);
  /// Remove the block arguments by inserting phi-breaking copies.
  void removeBBArgs();
  /// Insert phi-breaking copies for the block argument.
  void handlePhiNode(BlockArgument bbArg, ArrayRef<Block *> blocks,
                     ArrayRef<OpOperand *> operands);
  IRRewriter &rewriter;
  DominanceInfo &domInfo;
  Block *entryBlock;
  std::set<std::pair<BlockArgument, OpOperand *>, CmpCFPair> controlFlowPairs;
};
} // namespace

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

/// Helper to deallocate an allocation.
static Value deallocAllocation(IRRewriter &rewriter, Value value) {
  auto regTy = cast<RegisterTypeInterface>(value.getType());
  if (!regTy || !regTy.hasUnallocatedSemantics())
    return value;
  bool isRange = regTy.isRegisterRange();
  if (isRange) {
    auto splitOp =
        SplitRegisterRangeOp::create(rewriter, value.getLoc(), value);
    SmallVector<Value> results;
    for (Value input : splitOp.getResults())
      results.push_back(DeallocCastOp::create(rewriter, value.getLoc(), input));
    return MakeRegisterRangeOp::create(rewriter, value.getLoc(), results);
  }
  return DeallocCastOp::create(rewriter, value.getLoc(), value);
}

/// Insert copies for values that would be clobbered by reused allocas.
///
/// When the same alloca is used as outs for multiple instructions, later
/// instructions could clobber earlier values. If an earlier value is still
/// live at the time the clobbering instruction is executed, we must copy it
/// first.
///
/// Example:
///   %0 = alloca
///   %1 = test_inst outs %0          // %1 stored in %0
///   %2 = test_inst outs %0          // CLOBBERS %1!
///   test_inst ins %1, %2            // uses both - %1 is garbage
/// TODO: Consider using liveness to avoid inserting copies for values that are
/// not clobbered. Note, that this is not really an issue as the inserted copy
/// operation is pure, so it will DCE'd if unused.
static void removePotentiallyClobberedValues(Operation *op,
                                             IRRewriter &rewriter,
                                             DominanceInfo &domInfo) {
  op->walk([&](InstOpInterface inst) {
    LDBG() << "Handling clobbered values for: " << inst;
    rewriter.setInsertionPoint(inst);
    for (auto [i, out] : llvm::enumerate(inst.getInstOuts())) {
      auto regTy = cast<RegisterTypeInterface>(out.getType());
      if (!regTy.hasValueSemantics())
        continue;
      LDBG() << "  Handling potentially clobbered output operand: " << i;
      auto alloc = createAllocation(rewriter, inst->getLoc(), regTy);
      auto cpy = lsir::CopyOp::create(rewriter, inst->getLoc(), alloc, out);
      rewriter.replaceUsesWithIf(out, cpy, [&](OpOperand &use) -> bool {
        return domInfo.properlyDominates(inst, use.getOwner());
      });
      if (mlir::isOpTriviallyDead(cpy)) {
        rewriter.eraseOp(cpy);
        rewriter.eraseOp(alloc.getDefiningOp());
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// CFGSimplifier
//===----------------------------------------------------------------------===//
void CFGSimplifier::run(Region &region) {
  {
    DenseSet<void *> irreducibleElements;
    DenseSet<BlockArgument> visitedBBArgs;
    LDBG() << "Computing control-flow pairs for: " << region;
    for (Block &block : region.getBlocks())
      computeControlFlowPairs(&block, irreducibleElements, visitedBBArgs);

    // Add to the irreducibleElements set the block arguments that are not in
    // the visitedBBArgs set (at this point this match exactly those in the
    // control-flow pairs).
    for (Block &block : region.getBlocks()) {
      for (BlockArgument arg : block.getArguments()) {
        if (!visitedBBArgs.contains(arg))
          irreducibleElements.insert(arg.getAsOpaquePointer());
      }
    }
    LDBG() << "Reducing control-flow pairs";
    reduceControlFlowPairs(irreducibleElements);
  }

  // Remove the block arguments by inserting phi-breaking copies.
  LDBG() << "Removing block arguments";
  removeBBArgs();
}

void CFGSimplifier::computeControlFlowPairs(
    Block *block, DenseSet<void *> &irreducibleElements,
    DenseSet<BlockArgument> &visitedBBArgs) {
  // Bail if the block is empty.
  if (block->empty())
    return;

  // Get the last operation and its successors.
  Operation *lastOp = &block->back();
  SuccessorRange succRange = lastOp->getSuccessors();

  // Bail if the last operation has no successors.
  if (succRange.empty())
    return;

  LDBG() << "  Handling terminator: " << *lastOp;

  // Bail if the last operation is not a branch.
  auto terminator = dyn_cast<BranchOpInterface>(lastOp);
  if (!terminator) {
    LDBG() << "  Terminator is not a branch, adding operands to "
              "irreducibleElements set";
    // Add the last operation's operands to the irreducibleElements set.
    {
      MutableArrayRef<OpOperand> operands = lastOp->getOpOperands();
      irreducibleElements.reserve(irreducibleElements.size() + operands.size());
      for (OpOperand &operand : operands)
        irreducibleElements.insert(&operand);
    }

    // Add the successors' block arguments to the irreducibleElements set.
    for (Block *succBB : succRange) {
      ValueRange succArgs = succBB->getArguments();
      irreducibleElements.reserve(irreducibleElements.size() + succArgs.size());
      for (Value arg : succArgs)
        irreducibleElements.insert(arg.getAsOpaquePointer());
    }
    return;
  }

  // Iterate over the successors to compute the control-flow pairs.
  for (auto [i, succ] : llvm::enumerate(succRange)) {
    LDBG() << "  Handling successor: " << i;
    // Get the successor operands and block arguments.
    SuccessorOperands operands = terminator.getSuccessorOperands(i);
    MutableArrayRef<BlockArgument> bbArgs = succ->getArguments();

    // Add the hidden block arguments to the irreducibleElements set.
    int32_t numHiddenArgs = operands.getProducedOperandCount();
    irreducibleElements.reserve(irreducibleElements.size() + numHiddenArgs);
    for (int32_t a = 0; a < numHiddenArgs; ++a)
      irreducibleElements.insert(bbArgs[a].getAsOpaquePointer());

    // Add the control-flow pairs to the controlFlowPairs set.
    for (auto [operand, bbArg] :
         llvm::zip_equal(operands.getMutableForwardedOperands(),
                         bbArgs.drop_front(numHiddenArgs))) {
      // Bail if the operand or block argument is in the irreducibleElements
      // set.
      bool invalidOperand = irreducibleElements.contains(&operand);
      bool invalidBBArg =
          irreducibleElements.contains(bbArg.getAsOpaquePointer());
      LDBG() << "  Handling control-flow pair: " << operand.getOperandNumber()
             << " -> " << bbArg;
      LDBG() << "    Invalid operand: " << invalidOperand;
      LDBG() << "    Invalid BB arg: " << invalidBBArg;
      if (invalidOperand || invalidBBArg) {
        if (invalidOperand)
          irreducibleElements.insert(bbArg.getAsOpaquePointer());
        if (invalidBBArg)
          irreducibleElements.insert(&operand);
        continue;
      }

      LDBG() << "  Adding control-flow pair to set";
      // Update the visited elements and control-flow pairs.
      visitedBBArgs.insert(bbArg);
      controlFlowPairs.insert({bbArg, &operand});
    }
  }
}

void CFGSimplifier::reduceControlFlowPairs(
    DenseSet<void *> &irreducibleElements) {
  // Iterate over the controlFlowPairs set to remove the irreducible operands
  // and block arguments.
  SmallVector<std::pair<BlockArgument, OpOperand *>> toRemove;

  // Iterate until there are no more changes.
  while (!controlFlowPairs.empty()) {
    toRemove.clear();
    for (auto [bbArg, operand] : controlFlowPairs) {
      // Mark for deletion if the block argument is in the irreducibleElements
      // set.
      if (irreducibleElements.contains(bbArg.getAsOpaquePointer())) {
        toRemove.push_back({bbArg, operand});
        irreducibleElements.insert(operand);
        continue;
      }

      // Mark for deletion if the operand is in the irreducibleElements set.
      if (irreducibleElements.contains(operand)) {
        toRemove.push_back({bbArg, operand});
        irreducibleElements.insert(bbArg.getAsOpaquePointer());
      }
    }
    // Stop if there were no more changes.
    if (toRemove.empty())
      return;

    // Remove the invalid operands and block arguments.
    for (auto elem : toRemove)
      controlFlowPairs.erase(elem);
  }
}

void CFGSimplifier::removeBBArgs() {
  SetVector<Block *> blocks;
  SetVector<OpOperand *> operands;
  DenseSet<OpOperand *> operandsToRemove;
  // Iterate until all control-flow pairs have been processed.
  while (!controlFlowPairs.empty()) {
    blocks.clear();
    operands.clear();
    // Find all the control-flow pairs with the same block argument.
    BlockArgument currentBBArg = nullptr;
    auto begin = controlFlowPairs.begin();
    auto it = begin;
    auto end = controlFlowPairs.end();
    currentBBArg = begin->first;
    LDBG() << "  Handling block argument: " << currentBBArg;

    // Iterate until the block argument changes.
    while (it != end) {
      if (it->first != currentBBArg)
        break;
      blocks.insert(it->second->getOwner()->getBlock());
      operands.insert(it->second);
      operandsToRemove.insert(it->second);
      ++it;
    }

    // Insert phi-breaking copies for the block argument.
    handlePhiNode(currentBBArg, blocks.getArrayRef(), operands.getArrayRef());

    // Remove the control-flow pairs.
    controlFlowPairs.erase(begin, it);
  }

  // Update the terminators by erasing the operands handled by the phi-breaking
  // copies.
  while (!operandsToRemove.empty()) {
    // Get the operand to remove.
    auto it = operandsToRemove.begin();
    OpOperand *operand = *it;
    operandsToRemove.erase(it);
    unsigned operandNumber = operand->getOperandNumber();

    // Get the branch operation.
    auto brOp = cast<BranchOpInterface>(operand->getOwner());
    LDBG() << "  Handling branch operation: " << brOp;
    // Iterate over the successors to find the successor that handles the
    // operand.
    // NOTE: This is highly inefficient, but it's the only safe way to do this
    // with the current interface.
    for (int32_t i = 0, e = brOp->getNumSuccessors(); i < e; ++i) {
      MutableOperandRange mutableOperands =
          brOp.getSuccessorOperands(i).getMutableForwardedOperands();
      // Bail if the successor operands are empty.
      if (mutableOperands.empty())
        continue;

      unsigned beginOperandIndex =
          mutableOperands.getAsOperandRange().getBeginOperandIndex();
      // Erase the operand if it is in the mutable operands range.
      if (operandNumber >= beginOperandIndex &&
          beginOperandIndex + mutableOperands.size() > operandNumber) {
        mutableOperands.erase(operandNumber - beginOperandIndex);

        // Break to avoid erasing the operand multiple times.
        break;
      }
    }
    LDBG() << "   Updated branch operation: " << brOp;
  }
}

/// Find the nearest common dominator for the given blocks.
/// This function was adapted from upstream, because upstream has a bug. TODO:
/// fix upstream.
static Block *findNearestCommonDominator(DominanceInfo &info,
                                         ArrayRef<Block *> blocks) {
  if (blocks.begin() == blocks.end())
    return nullptr;
  Block *dom = *blocks.begin();
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    dom = info.findNearestCommonDominator(dom, *it);
    if (!dom)
      return nullptr;
  }
  return dom;
}

void CFGSimplifier::handlePhiNode(BlockArgument bbArg, ArrayRef<Block *> blocks,
                                  ArrayRef<OpOperand *> operands) {
  // Bail if the block argument is not a register type with value semantics.
  auto regTy = dyn_cast<RegisterTypeInterface>(bbArg.getType());
  if (!regTy || !regTy.hasValueSemantics())
    return;

  // Get the nearest common dominator or the entry block if there is no common
  // dominator.
  Block *domBB = findNearestCommonDominator(domInfo, blocks);
  if (!domBB || domBB == bbArg.getOwner())
    domBB = entryBlock;

  LDBG() << "  Inserting phi-breaking copies for block argument: " << bbArg;
  // Create the new allocation.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(domBB->getTerminator());
  Value alloc =
      createAllocation(rewriter, bbArg.getLoc(), regTy.getAsUnallocated());

  // Insert the copy operations.
  llvm::SmallSet<std::tuple<void *, void *, void *>, 8> insertedCopies;
  for (OpOperand *operand : operands) {
    rewriter.setInsertionPoint(operand->getOwner());
    // Prevent inserting duplicate copies.
    if (!insertedCopies
             .insert({operand->getOwner()->getBlock(),
                      alloc.getAsOpaquePointer(),
                      operand->get().getAsOpaquePointer()})
             .second)
      continue;
    lsir::CopyOp::create(rewriter, bbArg.getLoc(), alloc, operand->get());
  }

  // Replace the block argument with the new allocation.
  rewriter.setInsertionPointToStart(bbArg.getOwner());
  auto newBBArg = deallocAllocation(rewriter, alloc);
  rewriter.replaceAllUsesWith(bbArg, newBBArg);
  bbArg.getOwner()->eraseArgument(bbArg.getArgNumber());
}

//===----------------------------------------------------------------------===//
// Bufferization pass
//===----------------------------------------------------------------------===//

void Bufferization::runOnOperation() {
  getOperation()->walk([&](FunctionOpInterface op) {
    if (op.empty())
      return WalkResult::skip();

    IRRewriter rewriter(&getContext());
    // Erase unreachable blocks.
    (void)mlir::eraseUnreachableBlocks(rewriter, op.getFunctionBody());

    auto &domInfo = getAnalysis<DominanceInfo>();

    // Insert copies to remove potentially clobbered values.
    removePotentiallyClobberedValues(op, rewriter, domInfo);

    // Simplify the control-flow graph.
    CFGSimplifier(rewriter, domInfo, &op.getFunctionBody().front())
        .run(op.getFunctionBody());

    // Eliminate common subexpressions to remove redundant copies.
    mlir::eliminateCommonSubExpressions(rewriter, domInfo, op);

    return WalkResult::skip();
  });
}
