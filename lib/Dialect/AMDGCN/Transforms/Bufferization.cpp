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

#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

} // namespace

/// Insert a copy instruction (s_mov_b32 or v_mov_b32_e32) based on type.
static Value insertCopy(OpBuilder &b, Location loc, Value out, Value v) {
  MLIRContext *ctx = b.getContext();
  if (auto sTy = dyn_cast<SGPRType>(out.getType())) {
    auto instAttr = InstAttr::get(ctx, OpCode::S_MOV_B32);
    return inst::SOP1Op::create(b, loc, sTy, instAttr, out, v);
  } else if (auto vTy = dyn_cast<VGPRType>(out.getType())) {
    auto instAttr = InstAttr::get(ctx, OpCode::V_MOV_B32_E32);
    return inst::VOP1Op::create(b, loc, vTy, instAttr, out, v);
  }
  assert(false && "expected SGPR or VGPR type");
  return nullptr;
}

/// Insert copies for phi-equivalent allocas that might interfere.
///
/// When multiple allocas flow to the same block argument, they are
/// "phi-equivalent" and traditionally share a register. However, if those
/// allocas interfere (both live at some point), they need separate registers.
///
/// Solution: Conservatively insert copy instructions before branches that pass
/// alloca-derived values to block arguments. The new allocas:
/// 1. are phi-equivalent (they flow to the same block arg)
/// 2. don't interfere by construction (they're in mutually exclusive branches)
///
/// Register allocation can then proceed, and DPS will attempt to reuse the same
/// register. When possible, this will result in self-copies that can easily be
/// eliminated post-hoc.
static void insertPhiBreakingCopies(Block *block, IRRewriter &rewriter,
                                    DataFlowSolver &solver,
                                    ValueProvenanceAnalysis *analysis) {
  for (BlockArgument arg : block->getArguments()) {
    auto regTy = dyn_cast<RegisterTypeInterface>(arg.getType());
    if (!regTy)
      continue;

    // Get the allocas that merge at this block arg.
    auto *lattice = solver.lookupState<dataflow::Lattice<ValueProvenance>>(arg);
    if (!lattice)
      continue;
    ArrayRef<Value> allocas = lattice->getValue().getAllocas();
    if (allocas.size() <= 1)
      continue;

    // Multiple allocas merge here - insert copies at each branch to break
    // interference.
    for (Block *pred : block->getPredecessors()) {
      auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
      if (!branchOp)
        continue;

      int64_t e = branchOp->getNumSuccessors();
      int64_t succIdx = 0;
      for (; succIdx < e; ++succIdx) {
        if (branchOp->getSuccessor(succIdx) == block)
          break;
      }
      assert(succIdx < e && "unexpected successor not found");

      Value operand =
          branchOp.getSuccessorOperands(succIdx)[arg.getArgNumber()];

      FailureOr<Value> provenance =
          analysis->getCanonicalPhiEquivalentAlloca(operand);
      if (failed(provenance))
        continue;

      // Insert copy: alloca + appropriate mov instruction
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(branchOp);
      auto operandTy = cast<RegisterTypeInterface>(operand.getType());
      Value out = AllocaOp::create(rewriter, branchOp->getLoc(), operandTy);
      Value copyResult = insertCopy(rewriter, branchOp->getLoc(), out, operand);

      // Update branch operand to use the copy result.
      branchOp.getSuccessorOperands(succIdx)
          .getMutableForwardedOperands()[arg.getArgNumber()]
          .set(copyResult);
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
///
/// Fix: Before %2's definition, copy %1 to a new alloca.
static void
removePotentiallyClobberedValues(Operation *op, IRRewriter &rewriter,
                                 DataFlowSolver &livenessSolver,
                                 ValueProvenanceAnalysis *provenanceAnalysis) {
  op->walk([&](InstOpInterface inst) {
    for (Value outsVal : inst.getInstOuts()) {
      FailureOr<Value> outsAlloca =
          provenanceAnalysis->getCanonicalPhiEquivalentAlloca(outsVal);
      if (failed(outsAlloca))
        continue;

      // Get liveness state before this instruction. Values that are live here
      // and originate from the same alloca will be clobbered by this write.
      const auto *liveness = livenessSolver.lookupState<LivenessState>(
          livenessSolver.getProgramPointBefore(inst));
      if (!liveness || liveness->isTop())
        continue;

      // Find live values from the same alloca - these will be clobbered.
      SmallVector<Value> clobberedValues;
      for (Value liveVal : liveness->getLiveValues()) {
        FailureOr<Value> liveAlloca =
            provenanceAnalysis->getCanonicalPhiEquivalentAlloca(liveVal);
        if (failed(liveAlloca) || *liveAlloca != *outsAlloca)
          continue;
        // Don't consider the outs value itself.
        if (liveVal == outsVal)
          continue;
        clobberedValues.push_back(liveVal);
      }

      // Insert copies for clobbered values.
      Block *instBlock = inst->getBlock();
      for (Value clobbered : clobberedValues) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(inst);
        auto copyAlloca =
            AllocaOp::create(rewriter, inst.getLoc(),
                             cast<RegisterTypeInterface>(clobbered.getType()));
        Value copyResult =
            insertCopy(rewriter, inst.getLoc(), copyAlloca, clobbered);

        // Replace "reading" uses of the clobbered value after inst (in same
        // block or in successor blocks) with the copy. Skip outs operands as
        // they do not read values.
        SmallPtrSet<Block *, 4> successorBlocks;
        for (Block *succ : instBlock->getSuccessors())
          successorBlocks.insert(succ);
        clobbered.replaceUsesWithIf(copyResult, [&](OpOperand &use) {
          Operation *useOwner = use.getOwner();
          Block *useBlock = useOwner->getBlock();
          // Allow same-block uses after the clobbering instruction.
          bool sameBlockAfter =
              useBlock == instBlock && inst->isBeforeInBlock(useOwner);
          // Allow uses in successor blocks (they execute after the clobber).
          // Exclude the instBlock itself from successors to avoid double-
          // counting with the same-block check (relevant for self-loops).
          bool inSuccessor =
              useBlock != instBlock && successorBlocks.contains(useBlock);
          if (!sameBlockAfter && !inSuccessor)
            return false;
          // Don't replace outs operands.
          if (auto userInst = dyn_cast<InstOpInterface>(useOwner)) {
            for (OpOperand &outsOp : userInst.getInstOutsMutable()) {
              if (&outsOp == &use)
                return false;
            }
          }
          return true;
        });
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Bufferization pass
//===----------------------------------------------------------------------===//

void Bufferization::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());

  // 1. Insert copies to resolve phi-equivalence conflicts.
  // Run value provenance analysis.
  DataFlowSolver provenanceSolver(DataFlowConfig().setInterprocedural(false));
  ValueProvenanceAnalysis *provenanceAnalysis =
      ValueProvenanceAnalysis::create(provenanceSolver, op);
  if (!provenanceAnalysis) {
    op->emitError() << "Failed to run value provenance analysis";
    return signalPassFailure();
  }

  // Insert copies to break interference between phi-equivalent allocas.
  op->walk([&](Block *block) {
    insertPhiBreakingCopies(block, rewriter, provenanceSolver,
                            provenanceAnalysis);
  });

  // 2. Insert copies to remove potentially clobbered values.
  // Fresh solver is required - the previous one retains stale phi-equivalence
  // state from before phi-breaking copies were inserted.
  DataFlowSolver freshProvenanceSolver(
      DataFlowConfig().setInterprocedural(false));
  provenanceAnalysis =
      ValueProvenanceAnalysis::create(freshProvenanceSolver, op);

  // Run liveness analysis for clobber detection.
  DataFlowSolver livenessSolver(DataFlowConfig().setInterprocedural(false));
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(livenessSolver);
  livenessSolver.load<LivenessAnalysis>(symbolTable, provenanceAnalysis);
  if (failed(livenessSolver.initializeAndRun(op))) {
    op->emitError() << "Failed to run liveness analysis";
    return signalPassFailure();
  }

  // Insert copies to remove potentially clobbered values.
  removePotentiallyClobberedValues(op, rewriter, livenessSolver,
                                   provenanceAnalysis);
}
