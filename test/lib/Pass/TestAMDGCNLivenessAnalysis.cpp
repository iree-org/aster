//===- TestAMDGCNLivenessAnalysis.cpp - Test AMDGCN Liveness Analysis -----===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for AMDGCN liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RegisterLiveness.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTAMDGCNLIVENESSANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestAMDGCNLivenessAnalysis pass
//===----------------------------------------------------------------------===//
class TestAMDGCNLivenessAnalysis
    : public mlir::aster::test::impl::TestAMDGCNLivenessAnalysisBase<
          TestAMDGCNLivenessAnalysis> {
public:
  using TestAMDGCNLivenessAnalysisBase::TestAMDGCNLivenessAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Create and configure the data flow solver
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    SymbolTableCollection symbolTable;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<RegisterLiveness>(symbolTable);

    // Initialize and run the solver
    if (failed(solver.initializeAndRun(op))) {
      op->emitError() << "failed to run buffer analysis";
      return signalPassFailure();
    }

    // Create the SSA map and print it.
    SSAMap ssaMap;
    ssaMap.populateMap(op);

    // Create the output stream and print the results.
    raw_prefixed_ostream os(llvm::outs(), "// ");
    os << "=== AMDGCN Liveness Analysis Results ===\n";
    os << "SSA map:\n";
    ssaMap.printMapMembers(os);
    os << "\n";

    // Walk through operations and print analysis results
    op->walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (auto symOp = dyn_cast<SymbolOpInterface>(operation))
        os << "Symbol: " << symOp.getName() << "\n";

      os << "Op: " << OpWithFlags(operation, OpPrintingFlags().skipRegions())
         << "\n";

      // Get the liveness state before and after this operation
      auto *beforeState = solver.lookupState<RegisterLivenessState>(
          solver.getProgramPointBefore(operation));
      auto *afterState = solver.lookupState<RegisterLivenessState>(
          solver.getProgramPointAfter(operation));

      os.indent();
      os << "LIVE BEFORE: ";
      if (beforeState)
        beforeState->print(os, ssaMap);
      else
        os << "<null>";
      os << "\n";

      os << "LIVE  AFTER: ";
      if (afterState)
        afterState->print(os, ssaMap);
      else
        os << "<null>";
      os << "\n";
      os.unindent();
    });
    os << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
