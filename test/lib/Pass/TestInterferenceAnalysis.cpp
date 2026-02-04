//===- TestInterferenceAnalysis.cpp - Test Interference Analysis ---------===//
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

#include "aster/Analysis/InterferenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTINTERFERENCEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestInterferenceAnalysis pass
//===----------------------------------------------------------------------===//
struct TestInterferenceAnalysis
    : public mlir::aster::test::impl::TestInterferenceAnalysisBase<
          TestInterferenceAnalysis> {
  using TestInterferenceAnalysisBase::TestInterferenceAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Walk through kernels and run analysis on each one.
    op->walk([&](KernelOp kernel) {
      llvm::errs() << "// Kernel: " << kernel.getSymName() << "\n";

      // Create the interference graph.
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      SymbolTableCollection symbolTable;
      FailureOr<InterferenceAnalysis> graph =
          InterferenceAnalysis::create(kernel, solver, symbolTable);
      if (failed(graph)) {
        kernel.emitError() << "Failed to build interference graph";
        return signalPassFailure();
      }

      graph->print(llvm::errs());
      llvm::errs() << "\n";
    });
  }
};
} // namespace
