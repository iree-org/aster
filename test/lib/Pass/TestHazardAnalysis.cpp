//===- TestHazardAnalysis.cpp - Test Hazard Analysis ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for hazard analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/HazardAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/HazardManager.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTHAZARDANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
class TestHazardAnalysis
    : public mlir::aster::test::impl::TestHazardAnalysisBase<
          TestHazardAnalysis> {
public:
  using TestHazardAnalysisBase::TestHazardAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Create hazard manager and data flow solver.
    // TODO: Don't hardcode CDNA3, make it configurable from module target.
    HazardManager hazardManager(op);
    hazardManager.populateHazardsFor(ISAVersion::CDNA3);
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    solver.load<HazardAnalysis>(hazardManager);

    // Initialize and run the solver
    if (failed(solver.initializeAndRun(op))) {
      op->emitError() << "failed to run hazard analysis";
      return signalPassFailure();
    }

    raw_prefixed_ostream os(llvm::outs(), "// ");
    os << "=== Hazard Analysis Results ===\n";
    auto &domInfo = getAnalysis<DominanceInfo>();

    // Walk through operations and print analysis results
    op->walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (auto symOp = dyn_cast<SymbolOpInterface>(operation))
        os << "Symbol: " << symOp.getName() << "\n";

      // Get the state after this operation
      auto *afterState = solver.lookupState<HazardState>(
          solver.getProgramPointAfter(operation));
      assert(afterState && "hazard state should not be null");
      os << "Op: " << OpWithFlags(operation, OpPrintingFlags().skipRegions())
         << "\n";
      os.indent();
      os << "HAZARD STATE AFTER: ";
      afterState->printDeterministic(os, domInfo);
      os.unindent();
      os << "\n";
    });
  }
};
} // namespace
