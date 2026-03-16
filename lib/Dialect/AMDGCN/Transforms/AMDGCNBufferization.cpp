//===- AMDGCNBufferization.cpp --------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <cstdint>
#include <tuple>

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
// BufferizationImpl
//===----------------------------------------------------------------------===//
/// Register bufferization implementation.
/// The handling of block arguments is inspired by:
///   Benoit Boissinot, Alain Darte, Fabrice Rastello, Benoît Dupont de
///   Dinechin, Christophe Guillon. Revisiting Out-of-SSA Translation for
///   Correctness, Code Quality, and Efficiency. [Research Report] 2008, pp.14.
///   ⟨inria-00349925v1⟩
struct BufferizationImpl {
  BufferizationImpl(DominanceInfo &domInfo, DPSAnalysis &dpsAnalysis,
                    DPSClobberingAnalysis &dpsLiveness)
      : domInfo(domInfo), dpsAnalysis(dpsAnalysis), dpsLiveness(dpsLiveness) {}

  /// Run the bufferization transform.
  void run(FunctionOpInterface op);

  /// Insert de-clobbering allocas for the given operation.
  void handleInstruction(IRRewriter &rewriter, InstOpInterface op);

  /// Insert phi-breaking copies for the given block argument.
  void handleBlockArgument(IRRewriter &rewriter, BlockArgument arg);

  /// Insert phi-forwards.
  void handlePhiForwards(IRRewriter &rewriter);

  /// Insert phi-forwards for a group of phi-forwards.
  void handlePhiForwardGroup(IRRewriter &rewriter, int64_t start, int64_t end);

  /// Remove register values from the terminators and the given blocks.
  void handleBlocksAndTerminators(IRRewriter &rewriter,
                                  ArrayRef<Block *> blocks);

  DominanceInfo &domInfo;
  /// The entry block of the function.
  Block *entryBlock = nullptr;
  /// The DPS analysis.
  DPSAnalysis &dpsAnalysis;
  /// The DPS liveness analysis.
  DPSClobberingAnalysis &dpsLiveness;
  /// The set of branch operations.
  DenseSet<BranchOpInterface> branchOps;
  /// The set of phi-node replacements.
  SmallVector<std::pair<BlockArgument, Value>> phiReplacements;
  /// A list containing, the branch operation to forward from, the successor
  /// index, the value to forward, the block to forward to, the allocation to
  /// use, and the argument number.
  SmallVector<
      std::tuple<BranchOpInterface, int32_t, Value, Block *, Value, int64_t>>
      phiForwards;
  /// A map from the processed block to a unique deterministic ID.
  DenseMap<Block *, int64_t> blockToId;
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//

void BufferizationImpl::run(FunctionOpInterface op) {
  entryBlock = &op.getFunctionBody().getBlocks().front();
  IRRewriter rewriter(op->getContext());
  // Insert de-clobbering allocas for all instructions that needed.
  op.walk([&](InstOpInterface op) {
    rewriter.setInsertionPoint(op);
    handleInstruction(rewriter, op);
  });

  SmallVector<Block *> blocksToUpdate;
  // Insert phi-breaking copies for all blocks that needed.
  op.walk([&](Block *block) {
    blockToId[block] = blockToId.size();
    rewriter.setInsertionPointToStart(block);
    bool needsUpdate = false;
    for (BlockArgument arg : block->getArguments()) {
      auto regTy = dyn_cast<RegisterTypeInterface>(arg.getType());
      if (!regTy || !regTy.hasValueSemantics())
        continue;
      handleBlockArgument(rewriter, arg);
      needsUpdate = true;
    }
    if (needsUpdate)
      blocksToUpdate.push_back(block);
  });
  handlePhiForwards(rewriter);
  handleBlocksAndTerminators(rewriter, blocksToUpdate);
}

void BufferizationImpl::handleInstruction(IRRewriter &rewriter,
                                          InstOpInterface instOp) {
  ResultRange results = instOp.getInstResults();
  if (results.empty())
    return;

  LDBG() << "- Handling instruction: " << instOp;
  rewriter.setInsertionPoint(instOp);

  OperandRange outs = instOp.getInstOuts();
  MutableArrayRef<OpOperand> operands =
      instOp->getOpOperands().slice(outs.getBeginOperandIndex(), outs.size());
  ArrayRef<bool> resultInfo = dpsLiveness.getClobberingInfo(instOp);
  assert(results.size() == resultInfo.size() &&
         "expected number of results to match clobbering info size");
  // `pos` tracks position within register-value-semantic outs only (not all
  // outs). resultInfo has one entry per value-semantic out, matching results.
  int64_t pos = 0;
  for (auto &&[idx, out] : llvm::enumerate(operands)) {
    auto regTy = dyn_cast<RegisterTypeInterface>(out.get().getType());
    if (!regTy || !regTy.hasValueSemantics())
      continue;

    if (!resultInfo[pos++])
      continue;

    Value newAlloca = createAllocation(rewriter, instOp.getLoc(), regTy);
    out.set(newAlloca);
    LDBG() << "-- De-clobbering out operand: " << idx;
  }
}

void BufferizationImpl::handleBlockArgument(IRRewriter &rewriter,
                                            BlockArgument arg) {
  const DPSAnalysis::ProvenanceSet *provenance = dpsAnalysis.getProvenance(arg);
  assert(provenance != nullptr && "block argument must have provenance");

  auto regTy = cast<RegisterTypeInterface>(arg.getType());
  Location loc = arg.getLoc();
  Block *block = arg.getOwner();

  // Insert allocas to handle the breakage of the phi-node.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);
  Value commonAlloc = createAllocation(rewriter, loc, regTy.getAsUnallocated());
  Value argAlloc = createAllocation(rewriter, loc, regTy);

  // Save the branch, value to forward, destination block and allocation for
  // each phi-node to handle later.
  for (auto [branchOp, value, index] : *provenance) {
    auto brOp = cast<BranchOpInterface>(branchOp);
    branchOps.insert(brOp);
    phiForwards.push_back(
        {brOp, index, value, block, commonAlloc, arg.getArgNumber()});
  }

  // Insert a copy at the first possible insertion point in the block, which is
  // either before the first use or before the terminator if there are no uses.
  rewriter.setInsertionPoint(block->getTerminator());
  {
    SmallVector<Block::iterator> possibleIps;
    // Get all the uses of the argument and add them if they are in the same
    // block.
    for (OpOperand &use : arg.getUses()) {
      if (use.getOwner()->getBlock() != block)
        continue;
      possibleIps.push_back(Block::iterator(use.getOwner()));
    }

    // Sort the possible insertion points based on dominance order, so that we
    // can insert the copy as close as possible to the first use.
    llvm::sort(possibleIps, [&](Block::iterator lhs, Block::iterator rhs) {
      return domInfo.properlyDominates(block, lhs, block, rhs);
    });

    // If there are possible insertion points, insert the copy before the first
    // one.
    if (!possibleIps.empty())
      rewriter.setInsertionPoint(block, possibleIps.front());
  }

  auto cpy = lsir::CopyOp::create(rewriter, loc, argAlloc, commonAlloc);
  phiReplacements.push_back({arg, cpy.getTargetRes()});
}

void BufferizationImpl::handlePhiForwards(IRRewriter &rewriter) {
  auto getCmpTuple = [&](const std::tuple<BranchOpInterface, int32_t, Value,
                                          Block *, Value, int64_t> &elem) {
    BranchOpInterface brOp = std::get<0>(elem);
    int32_t index = std::get<1>(elem);
    Block *block = std::get<3>(elem);
    int64_t argNum = std::get<5>(elem);
    return std::make_tuple(brOp.getOperation(), index, blockToId[block],
                           argNum);
  };

  // Sort the phiForwards by the branch operation, the successor index, the
  // block to forward to, and the argument number.
  llvm::sort(phiForwards,
             [&](const std::tuple<BranchOpInterface, int32_t, Value, Block *,
                                  Value, int64_t> &a,
                 const std::tuple<BranchOpInterface, int32_t, Value, Block *,
                                  Value, int64_t> &b) {
               return getCmpTuple(a) < getCmpTuple(b);
             });

  auto it = phiForwards.begin();
  while (it != phiForwards.end()) {
    // Get the iterator to the first element with a different branch operation
    // or block.
    auto nextIt = it;
    while (++nextIt != phiForwards.end() &&
           std::get<0>(*nextIt) == std::get<0>(*it) &&
           std::get<1>(*nextIt) == std::get<1>(*it)) {
    }
    // Insert phi-forwards for the group.
    handlePhiForwardGroup(rewriter, it - phiForwards.begin(),
                          nextIt - phiForwards.begin());
    it = nextIt;
  }
}

void BufferizationImpl::handlePhiForwardGroup(IRRewriter &rewriter,
                                              int64_t start, int64_t end) {
  // Get the origin block and region.
  Block *prdBlock = std::get<0>(phiForwards[start])->getBlock();
  Region *prdRegion = prdBlock->getParent();

  SuccessorOperands succOperands =
      std::get<0>(phiForwards[start])
          .getSuccessorOperands(std::get<1>(phiForwards[start]));

  SmallVector<Value> fwdValues = llvm::to_vector(
      llvm::make_filter_range(succOperands.getForwardedOperands(), [](Value v) {
        auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
        return !regTy || !regTy.hasValueSemantics();
      }));
  succOperands.getMutableForwardedOperands().clear();

  // Create a new block to insert the phi-forwards.
  Block *newBlock =
      rewriter.createBlock(prdRegion, ++Region::iterator(prdBlock));
  rewriter.setInsertionPointToEnd(newBlock);

  // Create copies and set the successors.
  for (int64_t i = start; i < end; ++i) {
    auto [brOp, index, value, block, alloc, argNum] = phiForwards[i];
    lsir::CopyOp::create(rewriter, alloc.getLoc(), alloc, value);
    brOp->setSuccessor(newBlock, index);
  }

  // Create a branch op to the block to forward to.
  cf::BranchOp::create(rewriter, std::get<0>(phiForwards[start])->getLoc(),
                       std::get<3>(phiForwards[start]), fwdValues);
}

void BufferizationImpl::handleBlocksAndTerminators(IRRewriter &rewriter,
                                                   ArrayRef<Block *> blocks) {
  auto isRegValType = [](Value value) {
    auto regTy = dyn_cast<RegisterTypeInterface>(value.getType());
    return regTy && regTy.hasValueSemantics();
  };

  // For each branch op, remove successor operands with register value
  // semantics.
  for (BranchOpInterface branchOp : branchOps) {
    for (auto [idx, succ] : llvm::enumerate(branchOp->getSuccessors())) {
      SuccessorOperands succOperands = branchOp.getSuccessorOperands(idx);
      assert(succOperands.getProducedOperandCount() == 0 &&
             "expected no produced operands");
      MutableOperandRange forwarded =
          succOperands.getMutableForwardedOperands();
      int64_t start = 0;
      while (start < forwarded.size()) {
        if (!isRegValType(forwarded[start].get())) {
          ++start;
          continue;
        }
        forwarded.erase(start);
      }
    }
  }

  // Replace the phi-nodes.
  for (auto [arg, value] : phiReplacements)
    rewriter.replaceAllUsesWith(arg, value);

  // Erase block arguments with register value semantics.
  for (Block *block : blocks)
    block->eraseArguments(isRegValType);
}

//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//

void AMDGCNBufferization::runOnOperation() {
  Operation *moduleOp = getOperation();
  auto &domInfo = getAnalysis<DominanceInfo>();

  // Create the dataflow solver and load the liveness analysis.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<LivenessAnalysis>(symbolTable);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(moduleOp))) {
    moduleOp->emitError() << "failed to run liveness analysis";
    return signalPassFailure();
  }

  // Walk through the functions and run the bufferization transform.
  WalkResult result = moduleOp->walk([&](FunctionOpInterface op) {
    if (op.empty())
      return WalkResult::skip();

    // Run the DPS analysis.
    FailureOr<DPSAnalysis> dpsResult = DPSAnalysis::create(op);
    if (failed(dpsResult)) {
      op->emitError() << "failed to run DPS analysis";
      return WalkResult::interrupt();
    }

    // Run the DPS liveness analysis.
    FailureOr<DPSClobberingAnalysis> livenessResult =
        DPSClobberingAnalysis::create(*dpsResult, solver, op);
    if (failed(livenessResult)) {
      op->emitError() << "failed to run DPS liveness analysis";
      return WalkResult::interrupt();
    }

    // Run the bufferization transform.
    BufferizationImpl impl(domInfo, *dpsResult, *livenessResult);
    impl.run(op);

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  // Run CSE to clean up any redundant copies inserted by bufferization.
  IRRewriter rewriter(moduleOp->getContext());
  mlir::eliminateCommonSubExpressions(rewriter, domInfo, moduleOp);

  // Set post-condition: no register-typed block arguments remain.
  if (auto kernelOp = dyn_cast<KernelOp>(moduleOp))
    kernelOp.addNormalForms(
        {NoRegisterBlockArgsAttr::get(moduleOp->getContext())});
}
