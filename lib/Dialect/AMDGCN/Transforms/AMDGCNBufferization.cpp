//===- AMDGCNBufferization.cpp --------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/PrintingUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-bufferization"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNBUFFERIZATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//
struct AMDGCNBufferization
    : public amdgcn::impl::AMDGCNBufferizationBase<AMDGCNBufferization> {
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
    return std::make_tuple(lhs.first.getOwner(), lhs.first.getArgNumber(),
                           lhs.second->getOwner(),
                           lhs.second->getOperandNumber()) <
           std::make_tuple(rhs.first.getOwner(), rhs.first.getArgNumber(),
                           rhs.second->getOwner(),
                           rhs.second->getOperandNumber());
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

/// Copy the given values and propagate the copies to the uses that are
/// dominated by the given dominance point. If the dominance point is null, use
/// the copy operation as the dominance point.
static void copyAndPropagateValues(ValueRange values, IRRewriter &rewriter,
                                   DominanceInfo &info, Operation *domPoint) {
  for (Value out : values) {
    auto regTy = dyn_cast<RegisterTypeInterface>(out.getType());
    if (!regTy || !regTy.hasValueSemantics())
      continue;
    LDBG() << "- Handling potentially clobbered value: " << out;
    auto alloc = createAllocation(rewriter, out.getLoc(), regTy);
    auto cpyOp = lsir::CopyOp::create(rewriter, out.getLoc(), alloc, out);
    rewriter.replaceUsesWithIf(out, cpyOp.getTargetRes(), [&](OpOperand &use) {
      return info.properlyDominates(domPoint ? domPoint : cpyOp.getOperation(),
                                    use.getOwner());
    });
    if (isOpTriviallyDead(cpyOp)) {
      rewriter.eraseOp(cpyOp);
      rewriter.eraseOp(alloc.getDefiningOp());
    }
  }
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
  op->walk([&](Operation *op) {
    if (auto inst = dyn_cast<InstOpInterface>(op)) {
      rewriter.setInsertionPoint(inst);
      LDBG() << "Handling clobbered values for: " << inst;
      copyAndPropagateValues(inst.getInstOuts(), rewriter, domInfo, inst);
      return;
    }
    if (auto brOp = dyn_cast<BranchOpInterface>(op)) {
      rewriter.setInsertionPoint(brOp);
      LDBG() << "Handling clobbered values for: " << brOp;
      copyAndPropagateValues(brOp->getOperands(), rewriter, domInfo, nullptr);
      return;
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

  LDBG() << "- Handling terminator: " << *lastOp;

  // Bail if the last operation is not a branch.
  auto terminator = dyn_cast<BranchOpInterface>(lastOp);
  if (!terminator) {
    LDBG() << "-- Terminator is not a branch, adding operands to "
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
    LDBG() << "-- Handling successor: " << i;
    // Get the successor operands and block arguments.
    SuccessorOperands operands = terminator.getSuccessorOperands(i);
    MutableArrayRef<BlockArgument> bbArgs = succ->getArguments();

    // Add the hidden block arguments to the irreducibleElements set.
    int32_t numHiddenArgs = operands.getProducedOperandCount();
    irreducibleElements.reserve(irreducibleElements.size() + numHiddenArgs);
    for (int32_t a = 0; a < numHiddenArgs; ++a) {
      LDBG() << "-- Adding hidden bbArg to irreducibleElements set: "
             << bbArgs[a];
      irreducibleElements.insert(bbArgs[a].getAsOpaquePointer());
    }

    // Add the control-flow pairs to the controlFlowPairs set.
    for (auto [operand, bbArg] :
         llvm::zip_equal(operands.getMutableForwardedOperands(),
                         bbArgs.drop_front(numHiddenArgs))) {
      // Bail if the operand or block argument is in the irreducibleElements
      // set.
      bool invalidOperand = irreducibleElements.contains(&operand);
      bool invalidBBArg =
          irreducibleElements.contains(bbArg.getAsOpaquePointer());
      LDBG() << "-- Handling control-flow pair: " << operand.getOperandNumber()
             << " -> " << bbArg;
      LDBG() << "--- Invalid operand: " << (invalidOperand ? "true" : "false");
      LDBG() << "--- Invalid BB arg: " << (invalidBBArg ? "true" : "false");
      if (invalidOperand || invalidBBArg) {
        if (invalidOperand)
          irreducibleElements.insert(bbArg.getAsOpaquePointer());
        if (invalidBBArg)
          irreducibleElements.insert(&operand);
        continue;
      }

      LDBG() << "-- Adding control-flow pair to set";
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
        LDBG() << "-- Removing control-flow pair due to irreducible BB arg: "
               << ValueWithFlags(bbArg, true) << ", " << operand->getOwner()
               << ": " << operand->getOperandNumber();
        toRemove.push_back({bbArg, operand});
        irreducibleElements.insert(operand);
        continue;
      }

      // Mark for deletion if the operand is in the irreducibleElements set.
      if (irreducibleElements.contains(operand)) {
        LDBG() << "-- Removing control-flow pair due to irreducible operand: "
               << ValueWithFlags(bbArg, true) << ", " << operand->getOwner()
               << ": " << operand->getOperandNumber();
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

/// Dump a control-flow pair.
static void dumpControlFlowPair(llvm::raw_ostream &os,
                                std::pair<BlockArgument, OpOperand *> pair) {
  BlockArgument bbArg = pair.first;
  OpOperand *operand = pair.second;
  Block *bb = bbArg.getOwner();
  Operation *op = operand->getOwner();
  os << "{ bb = "
     << BlockWithFlags(bb, BlockWithFlags::PrintMode::PrintAsQualifiedOperand)
     << ", arg = " << bbArg.getArgNumber()
     << ", op = " << OpWithFlags(op, OpPrintingFlags().skipRegions())
     << ", operand = " << operand->getOperandNumber() << "}";
}

/// Dump a set of control-flow pairs.
static void dumpControlFlowPairs(
    llvm::raw_ostream &os,
    const std::set<std::pair<BlockArgument, OpOperand *>, CmpCFPair> &pairs) {
  os << "Control-flow pairs:\n";
  llvm::interleave(
      pairs, os,
      [&](const std::pair<BlockArgument, OpOperand *> &pair) {
        os << "  ";
        dumpControlFlowPair(os, pair);
      },
      "\n");
}

/// Comparator for OpOperands that sorts by owner block, then by operand number
/// in descending order.
static bool cmpOperands(OpOperand *lhs, OpOperand *rhs) {
  if (lhs->getOwner() < rhs->getOwner())
    return true;
  if (lhs->getOwner() > rhs->getOwner())
    return false;
  return lhs->getOperandNumber() > rhs->getOperandNumber();
}

void CFGSimplifier::removeBBArgs() {
  SetVector<Block *> blocks;
  SetVector<OpOperand *> operands;
  // NOTE: We need to remove operands in descending order of operand number
  // to avoid invalidating the operand numbers of the remaining operands.
  std::set<OpOperand *, decltype(&cmpOperands)> operandsToRemove(&cmpOperands);
  LDBG_OS([&](raw_ostream &os) { dumpControlFlowPairs(os, controlFlowPairs); });
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
    LDBG() << "-- Handling block argument: " << currentBBArg;

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
    LDBG() << "-- Erasing operand " << operandNumber
           << " from branch operation: " << brOp;
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
        LDBG() << "--- Erased operand from successor " << i;
        // Break to avoid erasing the operand multiple times.
        break;
      }
    }
    LDBG() << "--- Updated branch operation: " << brOp;
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

  LDBG() << "--- Inserting phi-breaking copies for block argument: " << bbArg;
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
  LDBG() << "--- Erased block argument: " << bbArg;
  bbArg.getOwner()->eraseArgument(bbArg.getArgNumber());
}

//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//

void AMDGCNBufferization::runOnOperation() {
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
