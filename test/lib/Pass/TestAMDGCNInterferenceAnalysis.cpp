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
#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
      raw_prefixed_ostream os(llvm::outs(), "// ");
      os << "Function: " << kernel.getName() << "\n";

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
      auto definitionFilter = [](Operation *op) {
        return isa<amdgcn::LoadOp>(op);
      };
      solver.load<ReachingDefinitionsAnalysis>(definitionFilter);
      FailureOr<RegisterInterferenceGraph> graph =
          RegisterInterferenceGraph::create(kernel, solver, symbolTable,
                                            *rangeAnalysis, buildMode);
      if (failed(graph)) {
        kernel.emitError() << "failed to build interference graph";
        return signalPassFailure();
      }

      // Print the interference graph (and optionally the quotient after
      // running optimizeGraph).
      std::optional<llvm::EquivalenceClasses<int32_t>> eqClasses;
      if (optimize)
        eqClasses = optimizeGraph(kernel, *graph, solver);
      if (eqClasses) {
        graph->print(os);
        os << "\n";
        // Print equivalence classes sorted by smallest member.
        int64_t numNodes = graph->sizeNodes();
        DenseSet<int32_t> printed;
        os << "EquivalenceClasses {\n";
        for (int32_t i = 0; i < numNodes; ++i) {
          int32_t leader = eqClasses->getLeaderValue(i);
          if (!printed.insert(leader).second)
            continue;
          SmallVector<int32_t> members(eqClasses->members(i));
          llvm::sort(members);
          os << "  [";
          llvm::interleave(members, os, [&](int32_t m) { os << m; }, ", ");
          os << "]\n";
        }
        os << "}\n";
        graph->print(os, *eqClasses);
      } else {
        graph->print(os);
      }
      os << "\n";
    });
  }
};
} // namespace
