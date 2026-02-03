//===- TestValueProvenanceAnalysis.cpp - Test Value Provenance Analysis ---===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for value provenance analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-value-provenance-analysis"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTVALUEPROVENANCEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestValueProvenanceAnalysis pass
//===----------------------------------------------------------------------===//
class TestValueProvenanceAnalysis
    : public mlir::aster::test::impl::TestValueProvenanceAnalysisBase<
          TestValueProvenanceAnalysis> {
public:
  using TestValueProvenanceAnalysisBase::TestValueProvenanceAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    llvm::outs() << "=== Value Provenance Analysis Results ===\n";

    op->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";
      llvm::outs() << kernel << "\n";

      DataFlowSolver solver;
      auto *provenanceAnalysis =
          ValueProvenanceAnalysis::create(solver, kernel);
      if (!provenanceAnalysis) {
        kernel.emitError() << "Failed to run value provenance analysis";
        return;
      }

      llvm::outs() << "\nValue Provenances:\n";
      kernel.walk([&](Operation *operation) {
        for (Value result : operation->getResults()) {
          auto canonicalResult =
              provenanceAnalysis->getCanonicalPhiEquivalentAlloca(result);
          llvm::outs() << "  ";
          result.printAsOperand(llvm::outs(), OpPrintingFlags());
          llvm::outs() << " -> ";
          if (succeeded(canonicalResult))
            canonicalResult->printAsOperand(llvm::outs(), OpPrintingFlags());
          else
            llvm::outs() << "<unknown>";
          llvm::outs() << "\n";
        }
      });

      llvm::outs() << "\nPhi-Equivalences:\n";
      llvm::SmallPtrSet<Value, 8> printedCanonicals;
      kernel.walk([&](Operation *operation) {
        if (!isa<AllocaOp>(operation))
          return;
        Value alloca = operation->getResult(0);
        auto canonicalResult =
            provenanceAnalysis->getCanonicalPhiEquivalentAlloca(alloca);
        if (failed(canonicalResult) ||
            !printedCanonicals.insert(*canonicalResult).second)
          return;

        auto equivalents = provenanceAnalysis->getPhiEquivalentAllocas(alloca);
        if (equivalents.size() > 1) {
          llvm::outs() << "  Phi-Equivalent: [";
          llvm::interleaveComma(equivalents, llvm::outs(), [](Value v) {
            v.printAsOperand(llvm::outs(), OpPrintingFlags());
          });
          llvm::outs() << "]\n";
        }
      });
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }
};
} // namespace
