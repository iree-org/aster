//===- LegalizeCF.cpp - Legalize CF ops to AMDGCN instructions ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes CF dialect operations (cf.cond_br, cf.br) to AMDGCN
// scalar branch instructions. It runs after register allocation when operands
// are in physical registers.
//
// The pass expects cf.cond_br conditions to come from amdgcn.is_cc (which
// tests an SCC or VCC register). The transformation:
//   - cf.cond_br (cond from amdgcn.is_cc) -> s_cbranch_scc1/scc0 or
//     s_cbranch_vccnz/vccz + s_branch
//   - cf.br -> s_branch
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

  /// Lower lsir.select with a register condition (SCC or VCC).
  LogicalResult lowerSelect(lsir::SelectOp selectOp);
};

LogicalResult LegalizeCF::lowerCondBranch(lsir::CondBranchOp condBr) {
  // The condition is a register type (SCC or VCC) directly.
  Value flagReg = condBr.getCondition();
  bool isVector = isa<VCCType>(flagReg.getType());

  // Note: We just drop block arguments as they are allocated and all values
  // flow through side effects.
  // TODO: In the future, this is better done as a RA legalization once we have
  // a side-effecting representation of instructions without return values.
  for (auto &brOpRange :
       {condBr.getTrueDestOperands(), condBr.getFalseDestOperands()}) {
    for (Value operand : brOpRange) {
      Type type = operand.getType();
      if (!isa<SGPRType, VGPRType>(type)) {
        return condBr.emitError()
               << "lsir.cond_br operand must have an allocated register type";
      }
    }
  }

  IRRewriter rewriter(condBr);
  rewriter.setInsertionPoint(condBr);

  // Create conditional branch based on which destination is the next physical
  // block. The fallthrough target must be the next block.
  Location loc = condBr.getLoc();
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();
  Block *currentBlock = condBr->getBlock();
  Block *nextBlock = currentBlock->getNextNode();

  // lsir.cond_br branches to trueDest when the condition register is nonzero.
  // Select branch opcodes based on whether the flag register is SCC or VCC.
  OpCode branchIfTrue =
      isVector ? OpCode::S_CBRANCH_VCCNZ : OpCode::S_CBRANCH_SCC1;
  OpCode branchIfFalse =
      isVector ? OpCode::S_CBRANCH_VCCZ : OpCode::S_CBRANCH_SCC0;

  // amdgcn::CBranchOp takes a label; later, the actual 16-bit PC-relative
  // offset is computed by the LLVM assembler (MC layer) when it assembles this
  // text into binary machine code. This is happening outside of aster.
  if (falseDest == nextBlock) {
    // Branch to trueDest if condition true, fallthrough to falseDest.
    CBranchOp::create(rewriter, loc, branchIfTrue, flagReg, trueDest,
                      falseDest);
  } else if (trueDest == nextBlock) {
    // Branch to falseDest if condition false, fallthrough to trueDest.
    CBranchOp::create(rewriter, loc, branchIfFalse, flagReg, falseDest,
                      trueDest);
  } else {
    // TODO: neither destination is the next block, we need more sophisticated
    // logic to insert explicit branch and create a new block. For this to
    // happen we need to first stabilize reg-alloc output guarantees (i.e. the
    // BBarg erasure needs to happen in the absence of SSA values flowing).
    // For now, emit an error if we reach such a case. The current behavior is
    // enough to model `scf.for` loops.
    return condBr.emitError()
           << "neither cf.cond_br destination is the next physical block; "
           << "block reordering not yet implemented";
  }

  // Erase the original cf.cond_br
  rewriter.eraseOp(condBr);

  return success();
}

LogicalResult LegalizeCF::lowerBranch(lsir::BranchOp br) {
  // Note: We just drop block arguments as they are allocated and all values
  // flow through side effects.
  // TODO: In the future, this is better done as a RA legalization once we have
  // a side-effecting representation of instructions without return values.
  for (Value operand : br.getDestOperands()) {
    Type type = operand.getType();
    if (!isa<SGPRType, VGPRType>(type)) {
      return br.emitError()
             << "lsir.br operand must have an allocated register type";
    }
  }

  Location loc = br.getLoc();
  IRRewriter rewriter(br);
  rewriter.setInsertionPoint(br);

  // Create unconditional branch
  BranchOp::create(rewriter, loc, OpCode::S_BRANCH, br.getDest());

  // Erase the original cf.br
  rewriter.eraseOp(br);

  return success();
}

LogicalResult LegalizeCF::lowerSelect(lsir::SelectOp selectOp) {
  // The condition is always a register type (SCC or VCC) after codegen.
  Value flagReg = selectOp.getCondition();
  bool isVector = isa<VCCType>(flagReg.getType());

  Location loc = selectOp.getLoc();
  IRRewriter rewriter(selectOp);
  rewriter.setInsertionPoint(selectOp);

  Value dst = selectOp.getDst();
  if (isVector) {
    // v_cndmask_b32: vdst = VCC[lane] ? src1 : src0 (note: reversed order!)
    // src0 = false_value, src1 = true_value, src2 = VCC
    // VOP2 src0 accepts i32/SGPR/VGPR, but src1 must be VGPR.
    Value trueVal = selectOp.getTrueValue();
    Value falseVal = selectOp.getFalseValue();
    if (!isa<VGPRType>(trueVal.getType())) {
      // src1 is not a VGPR. Materialize it into dst (an allocated VGPR)
      // via v_mov_b32, then use dst as src1.
      V_MOV_B32_E32::create(rewriter, loc, dst, trueVal);
      trueVal = dst;
    }
    amdgcn::inst::VOP2Op::create(rewriter, loc, OpCode::V_CNDMASK_B32, dst,
                                 /*dst1=*/nullptr, falseVal, trueVal, flagReg);
  } else {
    // s_cselect_b32: sdst = SCC ? src0 : src1.
    // src0 = true_value (selected when SCC=1), src1 = false_value.
    amdgcn::inst::SOP2Op::create(rewriter, loc, OpCode::S_CSELECT_B32, dst,
                                 selectOp.getTrueValue(),
                                 selectOp.getFalseValue());
  }

  // Replace uses of the select result with the dst (which now holds the
  // s_cselect_b32 or v_cndmask_b32 result via side effect).
  if (selectOp->getNumResults() > 0)
    rewriter.replaceOp(selectOp, dst);
  else
    rewriter.eraseOp(selectOp);

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

  // Collect all operations to lower.
  SmallVector<lsir::SelectOp> selects;
  SmallVector<lsir::CondBranchOp> condBranches;
  SmallVector<lsir::BranchOp> branches;
  op->walk([&](Operation *innerOp) {
    if (auto selectOp = dyn_cast<lsir::SelectOp>(innerOp))
      selects.push_back(selectOp);
    else if (auto condBr = dyn_cast<lsir::CondBranchOp>(innerOp))
      condBranches.push_back(condBr);
    else if (auto br = dyn_cast<lsir::BranchOp>(innerOp))
      branches.push_back(br);
  });

  // Lower i1-conditioned selects (they reference amdgcn.is_cc).
  for (lsir::SelectOp selectOp : selects) {
    if (failed(lowerSelect(selectOp))) {
      signalPassFailure();
      return;
    }
  }

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
