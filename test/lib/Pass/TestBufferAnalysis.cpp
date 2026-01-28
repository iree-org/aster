//===- TestBufferAnalysis.cpp - Test Buffer Analysis ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for buffer analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/BufferAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTBUFFERANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestBufferAnalysis pass
//===----------------------------------------------------------------------===//
class TestBufferAnalysis
    : public mlir::aster::test::impl::TestBufferAnalysisBase<
          TestBufferAnalysis> {
public:
  using TestBufferAnalysisBase::TestBufferAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    llvm::outs() << "=== Buffer Analysis Results ===\n";

    // Create and configure the data flow solver
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    solver.load<BufferAnalysis>(domInfo);

    // Initialize and run the solver
    if (failed(solver.initializeAndRun(op))) {
      op->emitError() << "Failed to run buffer analysis";
      return signalPassFailure();
    }

    // Walk through operations and print analysis results
    op->walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (auto symOp = dyn_cast<SymbolOpInterface>(operation)) {
        llvm::outs() << "Symbol: " << symOp.getName() << "\n";
      }
      llvm::outs() << "Op: "
                   << OpWithFlags(operation, OpPrintingFlags().skipRegions())
                   << "\n";

      // Get the buffer state before and after this operation
      auto *beforeState = solver.lookupState<BufferState>(
          solver.getProgramPointBefore(operation));
      auto *afterState = solver.lookupState<BufferState>(
          solver.getProgramPointAfter(operation));

      llvm::outs() << "\tLIVE BEFORE: ";
      if (beforeState)
        beforeState->print(llvm::outs());
      else
        llvm::outs() << "<null>";
      llvm::outs() << "\n";

      llvm::outs() << "\tLIVE  AFTER: ";
      if (afterState)
        afterState->print(llvm::outs());
      else
        llvm::outs() << "<null>";
      llvm::outs() << "\n";
    });
    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
