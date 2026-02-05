//===- TestDPSAliasAnalysis.cpp - Test DPS Alias Analysis -----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for DPS alias analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-dps-alias-analysis"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTDPSALIASANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestDPSAliasAnalysis pass
//===----------------------------------------------------------------------===//
class TestDPSAliasAnalysis
    : public mlir::aster::test::impl::TestDPSAliasAnalysisBase<
          TestDPSAliasAnalysis> {
public:
  using TestDPSAliasAnalysisBase::TestDPSAliasAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== DPS Alias Analysis Results ===\n";

    // Walk through kernels and run analysis on each one
    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";

      // First print the whole kernel op in llvm::outs() so we can capture the
      // SSA values during FileCheck.
      llvm::outs() << kernel << "\n";

      // Run ValueProvenanceAnalysis first in a separate solver.
      DataFlowSolver provenanceSolver;
      auto *provenanceAnalysis =
          ValueProvenanceAnalysis::create(provenanceSolver, kernel);
      if (!provenanceAnalysis) {
        kernel.emitError() << "Failed to run provenance analysis";
        return;
      }

      // Load DPSAliasAnalysis with the computed provenance.
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      dataflow::loadBaselineAnalyses(solver);
      auto *aliasAnalysis = solver.load<DPSAliasAnalysis>(provenanceAnalysis);
      if (!aliasAnalysis || failed(solver.initializeAndRun(kernel))) {
        kernel.emitError() << "Failed to run DPS alias analysis";
        return;
      }

      bool hasConflicts = !aliasAnalysis->getConflictingValues().empty();
      llvm::outs() << "Ill-formed IR: " << (hasConflicts ? "yes" : "no")
                   << "\n";

      // Print values with CONFLICT state (DPS violations).
      if (!aliasAnalysis->getConflictingValues().empty()) {
        llvm::outs() << "Conflicting values (DPS violations): [";
        llvm::interleaveComma(
            aliasAnalysis->getConflictingValues(), llvm::outs(),
            [](Value v) { v.printAsOperand(llvm::outs(), OpPrintingFlags()); });
        llvm::outs() << "]\n";
      }

      // Print values with UNKNOWN state (conservative, not errors).
      if (!aliasAnalysis->getUnknownValues().empty()) {
        llvm::outs() << "Unknown values (conservative): [";
        llvm::interleaveComma(
            aliasAnalysis->getUnknownValues(), llvm::outs(),
            [](Value v) { v.printAsOperand(llvm::outs(), OpPrintingFlags()); });
        llvm::outs() << "]\n";
      }

      // Group values by equivalence class ID (only register types).
      llvm::DenseMap<EqClassID, llvm::SmallVector<Value>> eqClassToValues;
      kernel.walk([&](Operation *operation) {
        for (Value result : operation->getResults()) {
          if (!isa<RegisterTypeInterface>(result.getType()))
            continue;
          for (EqClassID eqClassId : aliasAnalysis->getEqClassIds(result))
            eqClassToValues[eqClassId].push_back(result);
        }
      });

      // Print equivalence classes
      llvm::outs() << "\nEquivalence Classes:\n";
      for (auto [idx, allocaValue] :
           llvm::enumerate(aliasAnalysis->getValues())) {
        llvm::outs() << "  EqClass " << idx << ": [";
        llvm::interleaveComma(eqClassToValues[idx], llvm::outs(), [](Value v) {
          v.printAsOperand(llvm::outs(), OpPrintingFlags());
        });
        llvm::outs() << "]\n";
      }
      llvm::outs() << "\n";
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
