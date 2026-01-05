//===- TestConstantMemoryOffsetAnalysis.cpp - Test Constant Offset Analysis
//===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for constant memory offset analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ConstantMemoryOffsetAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-constant-memory-offset-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestConstantMemoryOffsetAnalysis pass
//===----------------------------------------------------------------------===//
class TestConstantMemoryOffsetAnalysis
    : public PassWrapper<TestConstantMemoryOffsetAnalysis, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConstantMemoryOffsetAnalysis)

  StringRef getArgument() const final {
    return "test-constant-memory-offset-analysis";
  }
  StringRef getDescription() const final {
    return "Test pass for constant memory offset analysis";
  }

  TestConstantMemoryOffsetAnalysis() = default;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== Constant Memory Offset Analysis Results ===\n";

    // Walk through kernels and run analysis on each one
    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";

      // Create and configure the data flow solver for this kernel
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      dataflow::loadBaselineAnalyses(solver);
      solver.load<ConstantMemoryOffsetAnalysis>();

      // Initialize and run the solver on the kernel
      if (failed(solver.initializeAndRun(kernel))) {
        kernel.emitError() << "Failed to run constant memory offset analysis";
        return;
      }

      // Walk through operations in the kernel and print analysis results
      kernel.walk([&](Operation *operation) {
        if (isa<amdgcn::KernelOp>(operation))
          return;

        // TODO: MemoryOpInterface.
        // Check if this is a load or store operation
        bool isLoad = isa<amdgcn::inst::GlobalLoadOp, amdgcn::inst::SMEMLoadOp,
                          amdgcn::inst::DSReadOp>(operation);
        bool isStore =
            isa<amdgcn::inst::GlobalStoreOp, amdgcn::inst::SMEMStoreOp,
                amdgcn::inst::DSWriteOp>(operation);

        if (!isLoad && !isStore)
          return;

        // Get the offset operand
        Value offsetOperand;
        if (auto globalLoad = dyn_cast<amdgcn::inst::GlobalLoadOp>(operation)) {
          offsetOperand = globalLoad.getVgprOffset();
        } else if (auto globalStore =
                       dyn_cast<amdgcn::inst::GlobalStoreOp>(operation)) {
          offsetOperand = globalStore.getVgprOffset();
        } else if (auto smemLoad =
                       dyn_cast<amdgcn::inst::SMEMLoadOp>(operation)) {
          // Skip S_MEMTIME which doesn't have an address
          if (smemLoad.getOpcode() == amdgcn::OpCode::S_MEMTIME)
            return;
          // SMEM operations don't have VGPR offsets
          return;
        } else if (auto smemStore =
                       dyn_cast<amdgcn::inst::SMEMStoreOp>(operation)) {
          // SMEM operations don't have VGPR offsets
          return;
        } else if (auto dsRead = dyn_cast<amdgcn::inst::DSReadOp>(operation)) {
          offsetOperand = dsRead.getOffset();
        } else if (auto dsWrite =
                       dyn_cast<amdgcn::inst::DSWriteOp>(operation)) {
          offsetOperand = dsWrite.getOffset();
        }

        if (!offsetOperand)
          return;

        // Get the state before this operation (offset is an input)
        auto *beforeState = solver.lookupState<ConstantMemoryOffsetLattice>(
            solver.getProgramPointBefore(operation));

        if (!beforeState || beforeState->isTop())
          return;

        // Get constant offset info for the offset operand
        ConstantMemoryOffsetInfo info = beforeState->getInfo(offsetOperand);

        llvm::outs() << "\nOperation: " << *operation << "\n";
        llvm::outs() << "\tCONSTANT OFFSET: " << info.constantOffset << " ";
        if (info.hasBaseValue())
          info.baseValue->print(llvm::outs());
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
void registerTestConstantMemoryOffsetAnalysisPass() {
  PassRegistration<TestConstantMemoryOffsetAnalysis>();
}
} // namespace test
} // namespace aster
} // namespace mlir
