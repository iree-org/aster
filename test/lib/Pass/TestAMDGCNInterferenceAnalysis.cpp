//===- TestAMDGCNInterferenceAnalysis.cpp - Test Interference Analysis ----===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for register interference analysis.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTAMDGCNINTERFERENCEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestAMDGCNInterferenceAnalysis pass
//===----------------------------------------------------------------------===//
struct TestAMDGCNInterferenceAnalysis
    : public mlir::aster::test::impl::TestAMDGCNInterferenceAnalysisBase<
          TestAMDGCNInterferenceAnalysis> {
  using TestAMDGCNInterferenceAnalysisBase::TestAMDGCNInterferenceAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Parse build mode option.
    RegisterInterferenceGraph::BuildMode buildMode;
    if (this->buildMode == "full") {
      buildMode = RegisterInterferenceGraph::BuildMode::Full;
    } else if (this->buildMode == "minimal") {
      buildMode = RegisterInterferenceGraph::BuildMode::Minimal;
    } else {
      op->emitError() << "build-mode must be \"full\" or \"minimal\", got \""
                      << this->buildMode << "\"";
      return signalPassFailure();
    }

    // Walk through kernels and run analysis on each one.
    op->walk([&](FunctionOpInterface kernel) {
      llvm::outs() << "// Function: " << kernel.getName() << "\n";

      // Create the range constraint analysis.
      FailureOr<RangeConstraintAnalysis> rangeAnalysis =
          RangeConstraintAnalysis::create(kernel);
      if (failed(rangeAnalysis)) {
        kernel.emitError() << "failed to run range constraint analysis";
        return signalPassFailure();
      }

      // Create the interference graph.
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      SymbolTableCollection symbolTable;
      FailureOr<RegisterInterferenceGraph> graph =
          RegisterInterferenceGraph::create(kernel, solver, symbolTable,
                                            *rangeAnalysis, buildMode);
      if (failed(graph)) {
        kernel.emitError() << "Failed to build interference graph";
        return signalPassFailure();
      }

      // Print the interference graph.
      graph->print(llvm::outs());
      llvm::outs() << "\n";
    });
  }
};
} // namespace
