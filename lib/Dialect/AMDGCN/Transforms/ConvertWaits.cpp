//===- ConvertWaits.cpp - Convert wait ops to hardware instructions ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNCONVERTWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct AMDGCNConvertWaits
    : public mlir::aster::amdgcn::impl::AMDGCNConvertWaitsBase<
          AMDGCNConvertWaits> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void AMDGCNConvertWaits::runOnOperation() {
  Operation *op = getOperation();

  // Get dominance info for the analysis.
  auto &domInfo = getAnalysis<DominanceInfo>();

  // Create and configure the data flow solver.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<WaitAnalysis>(domInfo);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "failed to run wait analysis";
    return signalPassFailure();
  }

  // Rewrite wait operations based on analysis results.
  IRRewriter rewriter(op->getContext());
  op->walk([&](WaitOp waitOp) {
    const auto *afterState =
        solver.lookupState<WaitState>(solver.getProgramPointAfter(waitOp));
    assert(afterState &&
           "expected valid wait analysis states before and after wait op");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(waitOp);
    WaitCnt counts = afterState->waitOpInfo->counts;

    // If no wait is needed, erase the wait op.
    if (counts.getVmcnt() == TokenState::kMaxPosition &&
        counts.getLgkmcnt() == TokenState::kMaxPosition) {
      rewriter.eraseOp(waitOp);
      return;
    }

    // Replace the wait op with an inst::SWaitcntOp with the minimal counts
    // required after this wait op.
    // Note that `SWaitcntOp` defaults to max counts, so we only need to set the
    // counts that are less than max.
    auto newWait = rewriter.replaceOpWithNewOp<inst::SWaitcntOp>(waitOp);

    // The default builder of SWaitcntOp sets both counts to max, so we take the
    // minimum of the max allowed hardware value count and the required count.
    newWait.setVmcnt(
        std::min(static_cast<int32_t>(newWait.getVmcnt()), counts.getVmcnt()));
    newWait.setLgkmcnt(std::min(static_cast<int32_t>(newWait.getLgkmcnt()),
                                counts.getLgkmcnt()));
  });
}
