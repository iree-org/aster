//===- AMDGCNHazards.cpp - AMDGCN hazard analysis pass --------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/HazardAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster::amdgcn {
#define GEN_PASS_DEF_AMDGCNHAZARDS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace mlir::aster::amdgcn

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AMDGCNHazards pass
//===----------------------------------------------------------------------===//
struct AMDGCNHazards : public amdgcn::impl::AMDGCNHazardsBase<AMDGCNHazards> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNHazards pass
//===----------------------------------------------------------------------===//
void AMDGCNHazards::runOnOperation() {
  Operation *op = getOperation();

  // Run hazard analysis.
  CDNA3Hazards hazardManager(op);
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<HazardAnalysis>(hazardManager);
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "failed to run hazard analysis";
    return signalPassFailure();
  }

  IRRewriter rewriter(op->getContext());

  // Collect the operations that required nop instructions.
  SmallVector<std::pair<InstOpInterface, InstCounts>> instWithNops;
  WalkResult result = op->walk([&](AMDGCNInstOpInterface instOp) {
    auto *afterState =
        solver.lookupState<HazardState>(solver.getProgramPointAfter(instOp));
    if (!afterState)
      return WalkResult::interrupt();

    InstCounts instCounts = afterState->flushedInstCounts;

    // Early return if no nop instructions are required.
    if (instCounts.isInactive())
      return WalkResult::advance();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(instOp);

    Location loc = instOp.getLoc();
    for (int i = 0; i < instCounts.getNumVector(); ++i)
      inst::VOP1NopOp::create(rewriter, loc, OpCode::V_NOP);

    int8_t numScalarNops = instCounts.getNumScalar();

    // TODO: Don't hardcode 16 here, and get it from the NOP instruction.
    for (int32_t i = 0; i < numScalarNops; i += 16) {
      int32_t nops = std::min(16, numScalarNops - i);
      inst::SOPPOp::create(rewriter, loc, OpCode::S_NOP,
                           static_cast<uint16_t>(nops));
    }

    assert(instCounts.getNumDataShare() == 0 &&
           "data share nop not implemented");
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    op->emitError() << "failed to resolve hazards";
    return signalPassFailure();
  }
}
