//===- TestReachingDefinitions.cpp - Test Reaching Definitions -----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for reaching definitions analysis.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/SSAMap.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTREACHINGDEFINITIONS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestReachingDefinitions pass
//===----------------------------------------------------------------------===//
class TestReachingDefinitions
    : public mlir::aster::test::impl::TestReachingDefinitionsBase<
          TestReachingDefinitions> {
public:
  using TestReachingDefinitionsBase::TestReachingDefinitionsBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    auto loadFilter =
        +[](Operation *op) -> bool { return isa<amdgcn::LoadOp>(op); };

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(solver);

    auto analysisOrFailure = ReachingDefinitionsAnalysis::create(
        solver, op,
        onlyLoads ? loadFilter : llvm::function_ref<bool(Operation *)>());
    if (failed(analysisOrFailure)) {
      op->emitError() << "IR is not in DPS normal form";
      return signalPassFailure();
    }
    if (failed(solver.initializeAndRun(op))) {
      op->emitError() << "Failed to run reaching definitions analysis";
      return signalPassFailure();
    }

    raw_prefixed_ostream os(llvm::outs(), "// ");
    os << "=== Reaching Definitions Analysis Results ===\n";
    op->walk([&](FunctionOpInterface op) {
      os << "Function: " << op.getName() << "\n";
      os << "SSA map:\n";
      SSAMap ssaMap;
      ssaMap.populateMap(op);
      ssaMap.printMapMembers(os);
      op.walk([&](Operation *op) {
        os << "Op: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << "\n";
        auto *afterState = solver.lookupState<ReachingDefinitionsState>(
            solver.getProgramPointAfter(op));
        os.indent();
        os << "REACHING DEFS AFTER: ";
        if (afterState)
          afterState->print(os, ssaMap, domInfo);
        else
          os << "<null>";
        os << "\n";
        os.unindent();
      });
    });
    os << "=== End Analysis Results ===\n";
  }
};
} // namespace
