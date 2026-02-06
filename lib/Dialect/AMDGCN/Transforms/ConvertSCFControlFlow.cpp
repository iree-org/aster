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
// CF dialect operations with explicit basic block structure. The pass uses
// thread uniform analysis to ensure loops are uniform before conversion.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Analysis/ABIAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
  /// Convert a scf.for operation to CF dialect control flow.
  LogicalResult convertForOp(scf::ForOp forOp, const ABIAnalysis &abiAnalysis);

  /// Convert a scf.if operation to CF dialect control flow.
  LogicalResult convertIfOp(scf::IfOp ifOp, const ABIAnalysis &abiAnalysis);
};

LogicalResult
ConvertSCFControlFlow::convertForOp(scf::ForOp forOp,
                                    const ABIAnalysis &abiAnalysis) {
  Location loc = forOp.getLoc();
  IRRewriter rewriter(forOp);

  // Get the loop bounds and step.
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  Type ivType = forOp.getInductionVar().getType();

  // Check if a value is i32 or index_cast from i32.
  auto isI32OrCastFromI32 = [](Value v) {
    if (v.getType().isInteger(32))
      return true;
    if (v.getType().isIndex()) {
      if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
        return castOp.getIn().getType().isInteger(32);
    }
    return false;
  };

  // Only i32 (or index_cast from i32) bounds are supported.
  if (!isI32OrCastFromI32(lowerBound) || !isI32OrCastFromI32(upperBound) ||
      !isI32OrCastFromI32(step)) {
    return forOp.emitError()
           << "only i32 induction variables are supported in this conversion "
              "(bounds must be i32 or arith.index_cast from i32)";
  }

  // Check if the loop is thread-uniform.
  bool isUniform = abiAnalysis.isThreadUniform(lowerBound).value_or(false) &&
                   abiAnalysis.isThreadUniform(upperBound).value_or(false) &&
                   abiAnalysis.isThreadUniform(step).value_or(false);
  if (!isUniform) {
    return forOp.emitError()
           << "only thread-uniform loops are supported in this conversion";
  }

  // Get the yield op and its operands before modifying the body.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> yieldOperands(yieldOp.getOperands());

  // Create the basic blocks for the loop structure.
  Block *bbPre = forOp->getBlock();
  Block *bbEnd = rewriter.splitBlock(bbPre, std::next(forOp->getIterator()));
  Block *bbBody = rewriter.createBlock(bbEnd);

  // Add block arguments to bbBody for induction variable and iter_args.
  bbBody->addArgument(ivType, loc);
  for (Value iterArg : forOp.getRegionIterArgs())
    bbBody->addArgument(iterArg.getType(), loc);

  // Add block arguments to bbEnd for the loop results (iter_args types).
  for (Value result : forOp.getResults())
    bbEnd->addArgument(result.getType(), loc);

  // Create the initial comparison: lowerBound < upperBound.
  rewriter.setInsertionPoint(forOp);
  Value initCond = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, lowerBound, upperBound);

  // Build initial branch operands: [lowerBound, init_args...]
  SmallVector<Value> initBranchArgs = {lowerBound};
  initBranchArgs.append(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  // Initial conditional branch: if lb < ub, enter loop; else skip to end.
  cf::CondBranchOp::create(rewriter, loc, initCond, bbBody, initBranchArgs,
                           bbEnd, forOp.getInitArgs());

  // Build the body. Get block arguments for IV and iter_args.
  rewriter.eraseOp(forOp.getBody()->getTerminator());
  Value ivBlockArg = bbBody->getArgument(0);
  SmallVector<Value> iterArgBlockArgs;
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    iterArgBlockArgs.push_back(bbBody->getArgument(i + 1));

  // Build the mapping from original region args to block args.
  SmallVector<Value> bodyArgMapping = {ivBlockArg};
  bodyArgMapping.append(iterArgBlockArgs);

  // Inline the loop body into bbBody.
  rewriter.setInsertionPointToEnd(bbBody);
  rewriter.inlineBlockBefore(forOp.getBody(), bbBody, bbBody->end(),
                             bodyArgMapping);

  // Compute the next induction variable value.
  Value ivNext = arith::AddIOp::create(rewriter, loc, ivBlockArg, step);

  // Create the back-edge comparison: ivNext < upperBound.
  Value backEdgeCond = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, ivNext, upperBound);

  // Build back-edge operands: [ivNext, yield_values...]
  SmallVector<Value> backEdgeArgs = {ivNext};
  backEdgeArgs.append(yieldOperands);

  // Conditional branch: if ivNext < ub, continue loop; else exit.
  cf::CondBranchOp::create(rewriter, loc, backEdgeCond, bbBody, backEdgeArgs,
                           bbEnd, yieldOperands);

  // Replace forOp results with bbEnd's block arguments.
  rewriter.replaceOp(forOp, bbEnd->getArguments());
  return success();
}

LogicalResult
ConvertSCFControlFlow::convertIfOp(scf::IfOp ifOp,
                                   const ABIAnalysis &abiAnalysis) {
  Location loc = ifOp.getLoc();
  IRRewriter rewriter(ifOp);

  Value condition = ifOp.getCondition();

  // Check if the condition is thread-uniform.
  if (!abiAnalysis.isThreadUniform(condition).value_or(false)) {
    return ifOp.emitError()
           << "only thread-uniform conditions are supported in this conversion";
  }

  bool hasElse = !ifOp.getElseRegion().empty();

  // Capture yield operands and block pointers before modifying anything.
  Block *thenBlock = ifOp.thenBlock();
  auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
  SmallVector<Value> thenYieldOperands(thenYield.getOperands());
  rewriter.eraseOp(thenYield);

  Block *elseBlock = nullptr;
  SmallVector<Value> elseYieldOperands;
  if (hasElse) {
    elseBlock = ifOp.elseBlock();
    auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
    elseYieldOperands.assign(elseYield.getOperands().begin(),
                             elseYield.getOperands().end());
    rewriter.eraseOp(elseYield);
  }

  // Split the current block after the ifOp to create the merge block.
  Block *bbMerge =
      rewriter.splitBlock(ifOp->getBlock(), std::next(ifOp->getIterator()));
  Block *bbThen = rewriter.createBlock(bbMerge);
  Block *bbElse = hasElse ? rewriter.createBlock(bbMerge) : bbMerge;

  // Add block arguments to bbMerge for the if results.
  for (Value result : ifOp.getResults())
    bbMerge->addArgument(result.getType(), loc);

  // Create conditional branch: if cond, then block; else block (or merge).
  rewriter.setInsertionPoint(ifOp);
  cf::CondBranchOp::create(rewriter, loc, condition, bbThen, ValueRange(),
                           bbElse, ValueRange());

  // Inline then region into bbThen and branch to merge.
  rewriter.inlineBlockBefore(thenBlock, bbThen, bbThen->end());
  rewriter.setInsertionPointToEnd(bbThen);
  cf::BranchOp::create(rewriter, loc, bbMerge, thenYieldOperands);

  // Inline else region into bbElse and branch to merge.
  if (hasElse) {
    rewriter.inlineBlockBefore(elseBlock, bbElse, bbElse->end());
    rewriter.setInsertionPointToEnd(bbElse);
    cf::BranchOp::create(rewriter, loc, bbMerge, elseYieldOperands);
  }

  // Replace ifOp results with bbMerge's block arguments.
  rewriter.replaceOp(ifOp, bbMerge->getArguments());
  return success();
}

void ConvertSCFControlFlow::runOnOperation() {
  Operation *op = getOperation();

  // Get the ABI analysis which includes thread uniform analysis.
  auto &abiAnalysis = getAnalysis<aster::ABIAnalysis>();

  // Collect all SCF operations first to avoid modifying while iterating.
  // Walk is post-order (inner before outer), but we need top-down order
  // (outer before inner) so that converting an outer op inlines the body
  // while inner SCF ops remain intact for later conversion.
  SmallVector<Operation *> scfOps;
  op->walk([&](Operation *nestedOp) {
    if (isa<scf::ForOp, scf::IfOp>(nestedOp))
      scfOps.push_back(nestedOp);
  });
  std::reverse(scfOps.begin(), scfOps.end());

  // Convert each SCF operation.
  for (Operation *scfOp : scfOps) {
    LogicalResult result = success();
    if (auto forOp = dyn_cast<scf::ForOp>(scfOp))
      result = convertForOp(forOp, abiAnalysis);
    else if (auto ifOp = dyn_cast<scf::IfOp>(scfOp))
      result = convertIfOp(ifOp, abiAnalysis);
    if (failed(result)) {
      signalPassFailure();
      return;
    }
  }

  // Verify post-condition: every cf.cond_br must have at least one destination
  // that is the next physical block. This is required by the downstream
  // legalize-cf pass which lowers cf.cond_br to s_cbranch_scc + fallthrough.
  op->walk([&](cf::CondBranchOp condBr) {
    Block *currentBlock = condBr->getBlock();
    Block *nextBlock = currentBlock->getNextNode();
    if (condBr.getTrueDest() != nextBlock &&
        condBr.getFalseDest() != nextBlock) {
      condBr.emitError()
          << "cf.cond_br produced by SCF conversion has neither destination "
          << "as the next physical block; this would fail in legalize-cf "
          << "(AMDGCN requires one branch target to be the fallthrough block)";
      signalPassFailure();
    }
  });
}

} // namespace
