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
#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
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
#include "llvm/ADT/EquivalenceClasses.h"
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
        return isa<amdgcn::LoadOpInterface>(op);
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
      // running optimizeGraph). Only print the quotient when coalescing
      // actually merged at least two original nodes; otherwise the quotient
      // is isomorphic to the original and printing it would be noise.
      std::optional<CoalescingInfo> coalescingInfo;
      if (optimize) {
        FailureOr<CoalescingInfo> info =
            CoalescingInfo::optimizeGraph(kernel, *graph);
        if (failed(info)) {
          kernel.emitError("failed to optimize interference graph");
          return signalPassFailure();
        }
        int64_t numQuotient = static_cast<int64_t>(info->values.size());
        if (numQuotient > 0 && numQuotient < graph->sizeNodes())
          coalescingInfo = std::move(*info);
      }
      if (coalescingInfo) {
        graph->print(os);
        os << "\nRange constraints:";
        for (const RangeConstraint &range : rangeAnalysis->getRanges()) {
          int32_t startId = graph->getNodeId(range.allocations.front());
          int32_t endId = graph->getNodeId(range.allocations.back());
          assert(startId >= 0 && endId >= 0 &&
                 "allocations should be in the graph");
          os << "\n  [" << startId << "-" << endId
             << "] alignment = " << range.alignment;
        }
        os << "\n";
        // Print equivalence classes sorted by smallest member. Each class is
        // stored as a sorted slice in memberData;
        // memberData[memberOffsets[qid]] is the minimum member.
        int32_t numQuotient =
            static_cast<int32_t>(coalescingInfo->memberOffsets.size()) - 1;

        // Returns the minimum original node ID in the quotient class of node.
        auto getMinMember = [&](int32_t node) {
          return coalescingInfo->memberData
              [coalescingInfo->memberOffsets[coalescingInfo->nodeClass[node]]];
        };

        os << "EquivalenceClasses {\n";
        for (int32_t qid = 0; qid < numQuotient; ++qid) {
          os << "  [";
          llvm::interleave(
              llvm::ArrayRef(coalescingInfo->memberData)
                  .slice(coalescingInfo->memberOffsets[qid],
                         coalescingInfo->memberOffsets[qid + 1] -
                             coalescingInfo->memberOffsets[qid]),
              os, [&](int32_t m) { os << m; }, ", ");
          os << "]\n";
        }
        os << "}\n";

        // Reconstruct an EquivalenceClasses for graph->print, which requires
        // it for colouring nodes by class.
        llvm::EquivalenceClasses<int32_t> eqForPrint;
        for (int32_t qid = 0; qid < numQuotient; ++qid) {
          int32_t rep =
              coalescingInfo->memberData[coalescingInfo->memberOffsets[qid]];
          for (int32_t j = coalescingInfo->memberOffsets[qid],
                       end = coalescingInfo->memberOffsets[qid + 1];
               j < end; ++j)
            eqForPrint.unionSets(rep, coalescingInfo->memberData[j]);
        }
        graph->print(os, eqForPrint);
        os << "\nPost range constraints:";
        DenseSet<int32_t> printed;
        for (const RangeConstraint &range : rangeAnalysis->getRanges()) {
          int32_t startId =
              getMinMember(graph->getNodeId(range.allocations.front()));
          if (!printed.insert(startId).second)
            continue;
          int32_t endId =
              getMinMember(graph->getNodeId(range.allocations.back()));
          os << "\n  [" << startId << "-" << endId
             << "] alignment = " << range.alignment;
        }
        os << "\n";
      } else {
        graph->print(os);
      }
      os << "\n";
    });
  }
};
} // namespace
