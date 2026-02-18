//===- TestDPSLiveness.cpp - Test DPS Liveness Analysis -----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for DPS liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTDPSLIVENESS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// TestDPSLiveness pass
//===----------------------------------------------------------------------===//
class TestDPSLiveness
    : public mlir::aster::test::impl::TestDPSLivenessBase<TestDPSLiveness> {
public:
  using TestDPSLivenessBase::TestDPSLivenessBase;

  void runOnOperation() override {
    Operation *moduleOp = getOperation();

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    SymbolTableCollection symbolTable;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<LivenessAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(moduleOp))) {
      moduleOp->emitError() << "failed to run liveness analysis";
      return signalPassFailure();
    }

    moduleOp->walk([&](FunctionOpInterface op) {
      SSAMap ssaMap;
      ssaMap.populateMap(op);
      FailureOr<DPSAnalysis> analysis = DPSAnalysis::create(op);
      if (failed(analysis)) {
        op->emitError() << "failed to run DPS analysis";
        return signalPassFailure();
      }
      FailureOr<DPSLiveness> liveness =
          DPSLiveness::create(*analysis, solver, op);
      if (failed(liveness)) {
        op->emitError() << "failed to run DPS liveness analysis";
        return signalPassFailure();
      }
      raw_prefixed_ostream os(llvm::outs(), "// ");
      os << "function: " << op.getNameAttr() << "\n";
      ssaMap.printMapMembers(os);
      os << "\n";
      liveness->print(os, ssaMap);
      os << "\n";
    });
  }
};
} // namespace
