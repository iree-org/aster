//===- TestAMDGCNSchedGraph.cpp - Test SchedGraph printer -----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass that builds and prints the AMDGCN value-
// scheduler SchedGraph for each amdgcn.kernel. Used to verify the dependence
// edges added by GraphBuilder (SSA, wait/barrier, sync-fence, architectural
// register RAW/WAR, i1 serialization).
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/SchedInterfaces.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTAMDGCNSCHEDGRAPH
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct TestAMDGCNSchedGraph
    : public mlir::aster::test::impl::TestAMDGCNSchedGraphBase<
          TestAMDGCNSchedGraph> {
  using TestAMDGCNSchedGraphBase::TestAMDGCNSchedGraphBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    SchedAnalysis analysis(root, solver, domInfo, getAnalysisManager());

    auto schedAttr = ValueSchedulerAttr::get(&getContext());
    if (failed(schedAttr.initializeAnalyses(analysis))) {
      root->emitError() << "failed to initialize sched analyses";
      return signalPassFailure();
    }
    if (failed(solver.initializeAndRun(root))) {
      root->emitError() << "failed to run dataflow analyses";
      return signalPassFailure();
    }

    root->walk([&](amdgcn::KernelOp kernel) {
      Block &block = kernel.getBodyRegion().front();
      auto graph = schedAttr.createGraph(&block, analysis);
      if (failed(graph)) {
        kernel.emitError() << "failed to build SchedGraph";
        signalPassFailure();
        return;
      }

      const SchedGraph &g = *graph;
      auto nodePrinter = [&g](raw_ostream &os, const int32_t &nodeId) {
        Operation *op = g.getOp(nodeId);
        os << "label = \"" << nodeId << ": " << op->getName().stripDialect()
           << "\"";
      };
      auto edgePrinter = [&g](raw_ostream &os, const Graph::Edge &edge) {
        Operation *src = g.getOp(edge.first);
        Operation *tgt = g.getOp(edge.second);
        os << "label = \"" << src->getName().stripDialect() << " -> "
           << tgt->getName().stripDialect() << "\"";
      };

      llvm::errs() << "Kernel: @" << kernel.getSymName() << "\n";
      g.print(llvm::errs(), "SchedGraph", nodePrinter, edgePrinter);
      llvm::errs() << "\n\n";
    });
  }
};
} // namespace
