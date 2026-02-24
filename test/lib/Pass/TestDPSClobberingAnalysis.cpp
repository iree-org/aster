//===- TestDPSClobberingAnalysis.cpp - Test DPS Clobbering Analysis -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for DPS clobbering analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTDPSCLOBBERINGANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// TestDPSClobberingAnalysis pass
//===----------------------------------------------------------------------===//
class TestDPSClobberingAnalysis
    : public mlir::aster::test::impl::TestDPSClobberingAnalysisBase<
          TestDPSClobberingAnalysis> {
public:
  using TestDPSClobberingAnalysisBase::TestDPSClobberingAnalysisBase;

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
      FailureOr<DPSAnalysis> analysis = DPSAnalysis::create(op);
      if (failed(analysis)) {
        op->emitError() << "failed to run DPS analysis";
        return signalPassFailure();
      }
      FailureOr<DPSClobberingAnalysis> clobberingInfo =
          DPSClobberingAnalysis::create(*analysis, solver, op);
      if (failed(clobberingInfo)) {
        op->emitError() << "failed to run DPS clobbering analysis";
        return signalPassFailure();
      }
      raw_prefixed_ostream os(llvm::outs(), "// ");
      os << "function: @" << op.getName() << "\n";
      op.walk([&](InstOpInterface instOp) {
        ArrayRef<bool> clobberingInfoForInst =
            clobberingInfo->getClobberingInfo(instOp);
        if (clobberingInfoForInst.empty())
          return;
        os.indent();
        os << instOp << "\n";
        os.indent();
        os << "[";
        llvm::interleaveComma(clobberingInfoForInst, os,
                              [&](bool b) { os << (b ? "true" : "false"); });
        os << "]\n";
        os.unindent(4);
      });
    });
  }
};
} // namespace
