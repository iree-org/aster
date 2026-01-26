//===- TestWaitAnalysis.cpp - Test Wait Analysis --------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for wait analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
class TestWaitAnalysis : public PassWrapper<TestWaitAnalysis, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWaitAnalysis)
  StringRef getArgument() const final { return "test-wait-analysis"; }
  StringRef getDescription() const final {
    return "Test pass for wait analysis";
  }
  TestWaitAnalysis() = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    // Create and configure the data flow solver for this kernel
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    solver.load<WaitAnalysis>(domInfo);

    // Initialize and run the solver on the kernel
    if (failed(solver.initializeAndRun(op))) {
      op->emitError() << "failed to run wait analysis";
      return signalPassFailure();
    }

    Builder b(&getContext());
    llvm::SmallString<128> str;
    llvm::raw_svector_ostream stream(str);

    llvm::outs() << "=== Wait Analysis Results ===\n";

    // Helper to get wait state as attribute
    auto getAttr = [&](const WaitState *state) {
      if (!state)
        return b.getStringAttr("<NULL>");
      str.clear();
      state->print(stream);
      return b.getStringAttr(str.str());
    };

    // Walk through operations in the kernel and print analysis results
    op->walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (auto symOp = dyn_cast<SymbolOpInterface>(operation)) {
        llvm::outs() << "Symbol: " << symOp.getName() << "\n";
      }

      // Get the state before / after this operation
      auto *beforeState = solver.lookupState<WaitState>(
          solver.getProgramPointBefore(operation));
      auto *afterState =
          solver.lookupState<WaitState>(solver.getProgramPointAfter(operation));
      StringAttr beforeAttr = getAttr(beforeState);
      StringAttr afterAttr = getAttr(afterState);
      llvm::outs() << "Op: "
                   << OpWithFlags(operation, OpPrintingFlags().skipRegions())
                   << "\n";
      llvm::outs() << "\tWAIT STATE BEFORE: " << beforeAttr.getValue() << "\n";
      llvm::outs() << "\tWAIT STATE AFTER: " << afterAttr.getValue() << "\n";
      operation->setAttr("wait_analysis.before", beforeAttr);
      operation->setAttr("wait_analysis.after", afterAttr);
    });
  }
};
} // namespace

namespace mlir {
namespace aster {
namespace test {
void registerTestWaitAnalysisPass() { PassRegistration<TestWaitAnalysis>(); }
} // namespace test
} // namespace aster
} // namespace mlir
