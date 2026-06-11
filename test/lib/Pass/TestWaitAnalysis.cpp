//===- TestWaitAnalysis.cpp - Test Wait Analysis --------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for wait analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTWAITANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
class TestWaitAnalysis
    : public mlir::aster::test::impl::TestWaitAnalysisBase<TestWaitAnalysis> {
public:
  using TestWaitAnalysisBase::TestWaitAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    Builder b(&getContext());
    llvm::SmallString<128> str;
    llvm::raw_svector_ostream stream(str);

    llvm::outs() << "=== Wait Analysis Results ===\n";

    auto runOn = [&](Operation *root, WaitCounterModel model) -> LogicalResult {
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      dataflow::loadBaselineAnalyses(solver);
      loadWaitAnalysis(solver, domInfo, model);
      if (failed(solver.initializeAndRun(root))) {
        root->emitError() << "failed to run wait analysis";
        return failure();
      }

      auto getAttr = [&](const WaitState *state) {
        if (!state)
          return b.getStringAttr("<NULL>");
        str.clear();
        state->print(stream, model);
        return b.getStringAttr(str.str());
      };

      root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
        if (auto symOp = dyn_cast<SymbolOpInterface>(operation))
          llvm::outs() << "Symbol: " << symOp.getName() << "\n";
        auto *beforeState = solver.lookupState<WaitState>(
            solver.getProgramPointBefore(operation));
        auto *afterState = solver.lookupState<WaitState>(
            solver.getProgramPointAfter(operation));
        StringAttr beforeAttr = getAttr(beforeState);
        StringAttr afterAttr = getAttr(afterState);
        llvm::outs() << "Op: "
                     << OpWithFlags(operation, OpPrintingFlags().skipRegions())
                     << "\n";
        llvm::outs() << "\tWAIT STATE BEFORE: " << beforeAttr.getValue()
                     << "\n";
        llvm::outs() << "\tWAIT STATE AFTER: " << afterAttr.getValue() << "\n";
        operation->setAttr("wait_analysis.before", beforeAttr);
        operation->setAttr("wait_analysis.after", afterAttr);
      });

      return success();
    };

    if (auto amdMod = dyn_cast<amdgcn::ModuleOp>(op)) {
      if (failed(runOn(op, getWaitCounterModelForOp(op))))
        return signalPassFailure();
    } else {
      bool foundNested = false;
      op->walk([&](amdgcn::ModuleOp amdMod) {
        foundNested = true;
        if (failed(runOn(amdMod, getWaitCounterModelForOp(amdMod))))
          signalPassFailure();
      });
      // Only analyze the top-level op directly when there is no typed
      // amdgcn.module (the bare-func tests). With nested modules the runs above
      // already cover them, and re-running here would apply the CDNA3 fallback
      // model to a module of a different ISA.
      if (!foundNested && failed(runOn(op, getWaitCounterModelForOp(op))))
        return signalPassFailure();
    }
  }
};
} // namespace
