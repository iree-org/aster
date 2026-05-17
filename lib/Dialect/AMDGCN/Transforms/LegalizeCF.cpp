//===- LegalizeCF.cpp - Legalize LSIR branch ops to AMDGCN instructions ---===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes LSIR branch operations to AMDGCN scalar branch
// instructions. It runs after register allocation when operands are in
// physical registers. The transformations performed are:
//   - lsir.cond_br (SCC condition) -> s_cbranch_scc1/scc0
//   - lsir.cond_br (VCC condition) -> s_cbranch_vccnz/vccz
//   - lsir.br -> s_branch
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LEGALIZECF
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// LegalizeCF pass
//===----------------------------------------------------------------------===//

struct LegalizeCF : public amdgcn::impl::LegalizeCFBase<LegalizeCF> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Lower lsir.cond_br to AMDGCN scalar/vector branch instructions.
  /// The condition is a register type (SCC or VCC) directly.
  LogicalResult lowerCondBranch(lsir::CondBranchOp condBr);

  /// Lower lsir.br to s_branch.
  LogicalResult lowerBranch(lsir::BranchOp br);
};

LogicalResult LegalizeCF::lowerCondBranch(lsir::CondBranchOp condBr) {
  // The condition is a register type (SCC or VCC) directly.
  Value flagReg = condBr.getCondition();
  bool isVector = isa<VCCType>(flagReg.getType());

  // All values flow through side effects; operands must be allocated registers.
  for (auto &brOpRange :
       {condBr.getTrueDestOperands(), condBr.getFalseDestOperands()}) {
    for (Value operand : brOpRange) {
      Type type = operand.getType();
      if (!isa<SGPRType, VGPRType>(type))
        return condBr.emitError()
               << "lsir.cond_br operand must have an allocated register type";
    }
  }
  assert(condBr.getTrueDest()->getNumArguments() == 0 &&
         condBr.getFalseDest()->getNumArguments() == 0 &&
         "destination blocks must have no block arguments after reg-alloc");

  IRRewriter rewriter(condBr);
  rewriter.setInsertionPoint(condBr);

  // Create conditional branch based on which destination is the next physical
  // block. The fallthrough target must be the next block.
  Location loc = condBr.getLoc();
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();
  Block *currentBlock = condBr->getBlock();
  Block *nextBlock = currentBlock->getNextNode();

  if (falseDest == nextBlock) {
    // Branch to trueDest if condition true, fallthrough to falseDest.
    if (isVector)
      SCbranchVccnz::create(rewriter, loc, flagReg, trueDest, falseDest);
    else
      SCbranchScc1::create(rewriter, loc, flagReg, trueDest, falseDest);
  } else if (trueDest == nextBlock) {
    // Branch to falseDest if condition false, fallthrough to trueDest.
    if (isVector)
      SCbranchVccz::create(rewriter, loc, flagReg, falseDest, trueDest);
    else
      SCbranchScc0::create(rewriter, loc, flagReg, falseDest, trueDest);
  } else {
    // TODO: neither destination is the next block, we need more sophisticated
    // logic to insert explicit branch and create a new block. For this to
    // happen we need to first stabilize reg-alloc output guarantees (i.e. the
    // BBarg erasure needs to happen in the absence of SSA values flowing).
    // For now, emit an error if we reach such a case. The current behavior is
    // enough to model `scf.for` loops.
    return condBr.emitError()
           << "neither lsir.cond_br destination is the next physical block; "
           << "block reordering not yet implemented";
  }

  // Erase the original lsir.cond_br.
  rewriter.eraseOp(condBr);

  return success();
}

LogicalResult LegalizeCF::lowerBranch(lsir::BranchOp br) {
  // Block arguments are dropped; see lowerCondBranch for rationale.
  for (Value operand : br.getDestOperands()) {
    Type type = operand.getType();
    if (!isa<SGPRType, VGPRType>(type))
      return br.emitError()
             << "lsir.br operand must have an allocated register type";
  }
  assert(br.getDest()->getNumArguments() == 0 &&
         "destination block must have no block arguments after reg-alloc");

  Location loc = br.getLoc();
  IRRewriter rewriter(br);
  rewriter.setInsertionPoint(br);

  // Create unconditional branch.
  SBranch::create(rewriter, loc, br.getDest());

  // Erase the original lsir.br.
  rewriter.eraseOp(br);

  return success();
}

void LegalizeCF::runOnOperation() {
  Operation *op = getOperation();

  // Pre-condition: all registers must be allocated before CF legalization.
  if (auto kernelOp = dyn_cast<KernelOp>(op)) {
    if (!kernelOp.hasNormalForm(
            AllRegistersAllocatedAttr::get(op->getContext()))) {
      op->emitError() << "amdgcn-legalize-cf requires "
                         "#amdgcn.all_registers_allocated normal form";
      return signalPassFailure();
    }
  }

  // Collect all branch operations to lower.
  SmallVector<lsir::CondBranchOp> condBranches;
  SmallVector<lsir::BranchOp> branches;
  op->walk([&](Operation *innerOp) {
    if (auto condBr = dyn_cast<lsir::CondBranchOp>(innerOp))
      condBranches.push_back(condBr);
    else if (auto br = dyn_cast<lsir::BranchOp>(innerOp))
      branches.push_back(br);
  });

  // Lower conditional branches.
  for (lsir::CondBranchOp condBr : condBranches) {
    if (failed(lowerCondBranch(condBr))) {
      signalPassFailure();
      return;
    }
  }

  // Lower unconditional branches.
  for (lsir::BranchOp br : branches) {
    if (failed(lowerBranch(br))) {
      signalPassFailure();
      return;
    }
  }

  // Set post-condition: no CF branches remain.
  if (auto kernelOp = dyn_cast<KernelOp>(op))
    kernelOp.addNormalForms({NoCfBranchesAttr::get(op->getContext())});
}

} // namespace
