//===- TestRangeAnalysis.cpp - Test Range Analysis ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for range analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Analysis/RangeAnalysis.h"
#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-range-analysis"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTRANGEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestRangeAnalysis pass
//===----------------------------------------------------------------------===//
class TestRangeAnalysis
    : public mlir::aster::test::impl::TestRangeAnalysisBase<TestRangeAnalysis> {
public:
  using TestRangeAnalysisBase::TestRangeAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== Range Analysis Results ===\n";

    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";
      llvm::outs() << kernel << "\n";

      // Run ValueProvenanceAnalysis first.
      DataFlowSolver provenanceSolver;
      auto *provenanceAnalysis =
          ValueProvenanceAnalysis::create(provenanceSolver, kernel);
      if (!provenanceAnalysis) {
        kernel.emitError() << "Failed to run provenance analysis";
        return;
      }

      // Run DPSAliasAnalysis.
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      dataflow::loadBaselineAnalyses(solver);
      auto *aliasAnalysis = solver.load<DPSAliasAnalysis>(provenanceAnalysis);
      if (failed(solver.initializeAndRun(kernel))) {
        kernel.emitError() << "Failed to run DPS alias analysis";
        return;
      }

      // Run RangeAnalysis.
      RangeAnalysis rangeAnalysis =
          RangeAnalysis::create(kernel, aliasAnalysis);

      // Print satisfiability.
      llvm::outs() << "Satisfiable: "
                   << (rangeAnalysis.isSatisfiable() ? "yes" : "no") << "\n";

      // Print ranges.
      llvm::outs() << "\nRanges:\n";
      for (auto [idx, range] : llvm::enumerate(rangeAnalysis.getRanges())) {
        llvm::outs() << "  Range " << idx << ": [";
        llvm::interleaveComma(range.getEqClassIds(), llvm::outs());
        llvm::outs() << "] from " << range.getOp() << "\n";
      }

      // Print graph.
      llvm::outs() << "\nDependency Graph:\n";
      rangeAnalysis.getGraph().print(llvm::outs());

      // Print allocations if satisfiable.
      if (rangeAnalysis.isSatisfiable()) {
        llvm::outs() << "\nAllocations:\n";
        for (auto [idx, alloc] :
             llvm::enumerate(rangeAnalysis.getAllocations())) {
          llvm::outs() << "  Allocation " << idx << ": [";
          llvm::interleaveComma(alloc.getAllocatedEqClassIds(), llvm::outs());
          llvm::outs() << "] (alignment=" << alloc.getAlignment() << ")\n";
        }

        // Print equivalence class to allocation mapping.
        llvm::outs() << "\nEqClass -> Allocation:\n";
        for (auto [eqClassId, allocaValue] :
             llvm::enumerate(aliasAnalysis->getValues())) {
          if (auto *alloc = rangeAnalysis.lookupAllocation(eqClassId)) {
            llvm::outs() << "  EqClass " << eqClassId << " -> Allocation "
                         << alloc->startEqClassId() << "\n";
          }
        }
      }
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
