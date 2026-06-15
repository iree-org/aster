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

#include "Passes.h"

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
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

    std::optional<ISAVersion> isaVersion =
        llvm::StringSwitch<std::optional<ISAVersion>>(isa.getValue())
            .Case("cdna3", ISAVersion::CDNA3)
            .Case("cdna4", ISAVersion::CDNA4)
            .Case("gfx1250", ISAVersion::GFX12_50)
            .Default(std::nullopt);
    if (!isaVersion) {
      root->emitError() << "invalid isa '" << isa
                        << "'; expected cdna3, cdna4, or gfx1250";
      return signalPassFailure();
    }

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);
    SchedAnalysis analysis(root, solver, domInfo, getAnalysisManager(),
                           *isaVersion);

    // The interface methods are external models; dispatch through the
    // interface.
    auto schedGraphAttr = mlir::cast<SchedGraphAttrInterface>(
        ValueSchedulerAttr::get(&getContext()));
    if (failed(schedGraphAttr.initializeAnalyses(analysis))) {
      root->emitError() << "failed to initialize sched analyses";
      return signalPassFailure();
    }
    if (failed(solver.initializeAndRun(root))) {
      root->emitError() << "failed to run dataflow analyses";
      return signalPassFailure();
    }

    root->walk([&](amdgcn::KernelOp kernel) {
      // Graph every block, not just the entry: loop bodies (where multi-buffer
      // DS ops live) are non-entry blocks, so the memdep edges only appear in
      // a later block's graph. Blocks are labeled by their index in the region.
      int32_t blockIdx = 0;
      for (Block &block : kernel.getBodyRegion()) {
        auto graph = schedGraphAttr.createGraph(&block, analysis);
        if (failed(graph)) {
          kernel.emitError()
              << "failed to build SchedGraph for block " << blockIdx;
          signalPassFailure();
          return;
        }

        const SchedGraph &g = *graph;
        // Append the first test.* attr (if any) to disambiguate same-opname
        // nodes/edges -- e.g. multiple ds_write_b32 to different LDS buffers.
        auto testAttrSuffix = [](Operation *op) -> std::string {
          for (NamedAttribute attr : op->getAttrs())
            if (attr.getName().getValue().starts_with("test."))
              return "/" + attr.getName().getValue().str();
          return "";
        };
        auto nodePrinter = [&g, &testAttrSuffix](raw_ostream &os,
                                                 const int32_t &nodeId) {
          Operation *op = g.getOp(nodeId);
          os << "label = \"" << nodeId << ": " << op->getName().stripDialect()
             << testAttrSuffix(op) << "\"";
        };
        auto edgePrinter = [&g, &testAttrSuffix](raw_ostream &os,
                                                 const Graph::Edge &edge) {
          Operation *src = g.getOp(edge.first);
          Operation *tgt = g.getOp(edge.second);
          os << "label = \"" << src->getName().stripDialect()
             << testAttrSuffix(src) << " -> " << tgt->getName().stripDialect()
             << testAttrSuffix(tgt) << "\"";
        };

        llvm::errs() << "Kernel: @" << kernel.getSymName()
                     << " Block: " << blockIdx << "\n";
        g.print(llvm::errs(), "SchedGraph", nodePrinter, edgePrinter);
        llvm::errs() << "\n\n";
        ++blockIdx;
      }
    });
  }
};
} // namespace
