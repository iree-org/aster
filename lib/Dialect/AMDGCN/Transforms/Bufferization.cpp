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

#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/PatternMatch.h"
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

//===----------------------------------------------------------------------===//
// Bufferization pass
//===----------------------------------------------------------------------===//

void Bufferization::runOnOperation() {
  Operation *op = getOperation();

  // Run value provenance analysis.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  ValueProvenanceAnalysis *provenanceAnalysis =
      ValueProvenanceAnalysis::create(solver, op);
  if (!provenanceAnalysis) {
    op->emitError() << "Failed to run value provenance analysis";
    return signalPassFailure();
  }

  IRRewriter rewriter(op->getContext());

  // Insert copies to break interference between phi-equivalent allocas.
  op->walk([&](Block *block) {
    insertPhiBreakingCopies(block, rewriter, solver, provenanceAnalysis);
  });
}
