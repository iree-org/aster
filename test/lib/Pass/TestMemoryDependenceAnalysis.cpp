//===- TestMemoryDependenceAnalysis.cpp - Test Memory Dependence Analysis ===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for memory dependence analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-memory-dependence-analysis"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTMEMORYDEPENDENCEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestMemoryDependenceAnalysis pass
//===----------------------------------------------------------------------===//
class TestMemoryDependenceAnalysis
    : public mlir::aster::test::impl::TestMemoryDependenceAnalysisBase<
          TestMemoryDependenceAnalysis> {
public:
  using TestMemoryDependenceAnalysisBase::TestMemoryDependenceAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== Memory Dependence Analysis Results ===\n";

    // Walk through kernels and run analysis on each one
    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";

      // Create and configure the data flow solver for this kernel
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      dataflow::loadBaselineAnalyses(solver);
      // Use default flushAllMemoryOnExit=true to match test expectations
      solver.load<MemoryDependenceAnalysis>(true);

      // Initialize and run the solver on the kernel
      if (failed(solver.initializeAndRun(kernel))) {
        kernel.emitError() << "Failed to run memory dependence analysis";
        return;
      }

      // Walk through operations in the kernel and print analysis results
      kernel.walk([&](Operation *operation) {
        if (isa<amdgcn::KernelOp>(operation))
          return;

        llvm::outs() << "\nOperation: " << *operation << "\n";

        // Get the state before / after this operation.
        // The before state carries the pending memory operations before this
        // one. The after state carries the must flush before op memory
        // operations
        // **before** this one.
        auto *beforeState = solver.lookupState<MemoryDependenceLattice>(
            solver.getProgramPointBefore(operation));
        auto *afterState = solver.lookupState<MemoryDependenceLattice>(
            solver.getProgramPointAfter(operation));

        // Lambda to print test.* attributes
        auto printTestAttrs = [](Operation *op) {
          for (auto attr : op->getAttrs()) {
            if (attr.getName().getValue().starts_with("test.")) {
              llvm::outs() << attr.getName().getValue() << ", ";
            }
          }
        };

        // Print test.* attributes for current operation
        bool hasTestAttr = llvm::any_of(operation->getAttrs(), [](auto attr) {
          return attr.getName().getValue().starts_with("test.");
        });

        if (hasTestAttr) {
          llvm::outs() << "\tPENDING BEFORE: "
                       << beforeState->getPendingAfterOpCount() << ": ";
          for (const auto &loc : beforeState->getPendingAfterOp())
            printTestAttrs(loc.op);
          llvm::outs() << "\n";
          llvm::outs() << "\tMUST FLUSH NOW: "
                       << afterState->getMustFlushBeforeOpCount() << ": ";
          for (const auto &loc : afterState->getMustFlushBeforeOp())
            printTestAttrs(loc.op);
          llvm::outs() << "\n";
        }
      });
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
