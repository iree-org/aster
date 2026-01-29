//===- ConvertSCFControlFlow.cpp - SCF to AMDGCN control flow conversion --===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass that converts SCF control flow operations to
// AMDGCN control flow instructions. It uses thread uniform analysis to
// determine whether to emit scalar (s_cmp) or vector (v_cmp) compare
// instructions.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Analysis/ABIAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_CONVERTSCFCONTROLFLOW
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// ConvertSCFControlFlow pass
//===----------------------------------------------------------------------===//

struct ConvertSCFControlFlow
    : public amdgcn::impl::ConvertSCFControlFlowBase<ConvertSCFControlFlow> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Convert a scf.for operation to AMDGCN control flow.
  LogicalResult convertForOp(scf::ForOp forOp, const ABIAnalysis &abiAnalysis);

  /// Get the appropriate compare opcode based on uniformity.
  /// Returns the opcode for a less-than comparison.
  OpCode getCmpLtOpCode(bool isUniform) const;
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

OpCode ConvertSCFControlFlow::getCmpLtOpCode(bool isUniform) const {
  return OpCode::S_CMP_LT_I32;
}

LogicalResult
ConvertSCFControlFlow::convertForOp(scf::ForOp forOp,
                                    const ABIAnalysis &abiAnalysis) {
  Location loc = forOp.getLoc();
  IRRewriter rewriter(forOp);

  // Get the loop bounds and step.
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Bail out if the induction variable is not i32.
  Value inductionVar = forOp.getInductionVar();
  if (!inductionVar.getType().isInteger(32)) {
    return forOp.emitError()
           << "only i32 induction variables are supported in this conversion";
  }

  // Check if the induction variable is thread-uniform.
  // The loop is uniform if lower bound, upper bound, and step are all uniform.
  bool isLowerUniform = abiAnalysis.isThreadUniform(lowerBound).value_or(false);
  bool isUpperUniform = abiAnalysis.isThreadUniform(upperBound).value_or(false);
  bool isStepUniform = abiAnalysis.isThreadUniform(step).value_or(false);
  bool isLoopUniform = isLowerUniform && isUpperUniform && isStepUniform;

  if (!isLoopUniform) {
    return forOp.emitError()
           << "only thread-uniform loops are supported in this conversion";
  }

  // Bail out if the loop has iter_args (not supported yet).
  if (!forOp.getInitArgs().empty()) {
    return forOp.emitError()
           << "loops with iter_args are not supported in this conversion";
  }

  Type sgprType = SGPRType::get(rewriter.getContext(), Register());

  // Create the basic blocks for the loop structure.
  Block *bbPre = forOp->getBlock();
  Block *bbEnd = rewriter.splitBlock(bbPre, std::next(forOp->getIterator()));
  Block *bbBody = rewriter.createBlock(bbEnd);

  // Convert bounds and step to registers or constants.
  auto toRegOrConst = [&](Value value) -> Value {
    if (matchPattern(value, m_Constant()))
      return value;

    return aster::lsir::ToRegOp::create(rewriter, loc, sgprType, value);
  };
  rewriter.setInsertionPoint(forOp);
  lowerBound = toRegOrConst(lowerBound);
  upperBound = toRegOrConst(upperBound);
  step = toRegOrConst(step);

  // Create the SCC register to hold the comparison result.
  Type destType = SCCType::get(rewriter.getContext());
  Value scc = AllocaOp::create(rewriter, loc, destType);

  // Create an SGPR register to hold the induction variable.
  Value ivReg = AllocaOp::create(rewriter, loc, sgprType);
  ivReg = S_MOV_B32::create(rewriter, loc, ivReg, lowerBound);

  // Create the initial comparison: iv < upperBound.
  CmpIOp::create(rewriter, loc, getCmpLtOpCode(isLoopUniform), scc, ivReg,
                 upperBound);
  CBranchOp::create(rewriter, loc, OpCode::S_CBRANCH_SCC0, scc, bbEnd, bbBody);

  // Build the body.
  rewriter.eraseOp(forOp.getBody()->getTerminator());
  rewriter.setInsertionPointToStart(bbBody);
  inductionVar =
      lsir::FromRegOp::create(rewriter, loc, inductionVar.getType(), ivReg);

  // Inline the loop body into the new block.
  rewriter.setInsertionPointToEnd(bbBody);
  rewriter.inlineBlockBefore(forOp.getBody(), bbBody, bbBody->end(),
                             {inductionVar});

  // Update the induction variable in the body.
  Value ivNext =
      amdgcn::S_ADD_I32::create(rewriter, loc, ivReg, ivReg, step).getResult(0);

  // Create the end-of-body comparison: ivNext < upperBound.
  CmpIOp::create(rewriter, loc, getCmpLtOpCode(isLoopUniform), scc, ivNext,
                 upperBound);
  CBranchOp::create(rewriter, loc, OpCode::S_CBRANCH_SCC1, scc, bbBody, bbEnd);

  // Erase the original forOp.
  rewriter.eraseOp(forOp);
  return success();
}

void ConvertSCFControlFlow::runOnOperation() {
  Operation *op = getOperation();

  // Get the ABI analysis which includes thread uniform analysis.
  auto &abiAnalysis = getAnalysis<aster::ABIAnalysis>();

  // Collect all scf.for operations first to avoid modifying while iterating.
  SmallVector<scf::ForOp> forOps;
  op->walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  // Convert each scf.for operation.
  for (scf::ForOp forOp : forOps) {
    if (failed(convertForOp(forOp, abiAnalysis))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace
