//===- TestLivenessAnalysis.cpp - Test Liveness Analysis ------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-liveness-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestLivenessAnalysis pass
//===----------------------------------------------------------------------===//
class TestLivenessAnalysis
    : public PassWrapper<TestLivenessAnalysis, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessAnalysis)

  StringRef getArgument() const final { return "test-liveness-analysis"; }
  StringRef getDescription() const final {
    return "Test pass for liveness analysis";
  }

  TestLivenessAnalysis() = default;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== Liveness Analysis Results ===\n";

    // Walk through kernels and run analysis on each one
    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";

      // Create and configure the data flow solver for this kernel
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      SymbolTableCollection symbolTable;
      dataflow::loadBaselineAnalyses(solver);
      auto *aliasAnalysis = solver.load<DPSAliasAnalysis>();
      solver.load<LivenessAnalysis>(symbolTable, aliasAnalysis);

      // Initialize and run the solver on the kernel
      if (failed(solver.initializeAndRun(kernel))) {
        kernel.emitError() << "Failed to run liveness analysis";
        return;
      }

      // Collect all values per equivalence class
      llvm::DenseMap<EqClassID, llvm::SmallVector<Value>> eqClassToValues;
      kernel.walk([&](Operation *operation) {
        for (Value result : operation->getResults()) {
          for (EqClassID eqClassId : aliasAnalysis->getEqClassIds(result))
            eqClassToValues[eqClassId].push_back(result);
        }
      });

      // Print equivalence classes from alias analysis
      llvm::outs() << "\nEquivalence Classes:\n";
      for (auto [idx, allocaValue] :
           llvm::enumerate(aliasAnalysis->getValues())) {
        llvm::outs() << "  EqClass " << idx << ": [";
        llvm::interleaveComma(eqClassToValues[idx], llvm::outs(), [](Value v) {
          v.printAsOperand(llvm::outs(), OpPrintingFlags());
        });
        llvm::outs() << "]\n";
      }

      // Walk through operations in the kernel and print analysis results
      kernel.walk([&](Operation *operation) {
        if (isa<amdgcn::KernelOp>(operation))
          return;

        llvm::outs() << "\nOperation: " << *operation << "\n";

        // Get the liveness state before and after this operation
        auto *beforeState = solver.lookupState<LivenessState>(
            solver.getProgramPointBefore(operation));
        auto *afterState = solver.lookupState<LivenessState>(
            solver.getProgramPointAfter(operation));

        llvm::outs() << "\tLIVE  AFTER: ";
        if (afterState)
          afterState->print(llvm::outs());
        else
          llvm::outs() << "<null>";
        llvm::outs() << "\n";

        llvm::outs() << "\tLIVE BEFORE: ";
        if (beforeState)
          beforeState->print(llvm::outs());
        else
          llvm::outs() << "<null>";
        llvm::outs() << "\n";
      });
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace

namespace mlir {
namespace aster {
namespace test {
void registerTestLivenessAnalysisPass() {
  PassRegistration<TestLivenessAnalysis>();
}
} // namespace test
} // namespace aster
} // namespace mlir
