//===- TestDPSAnalysis.cpp - Test DPS Analysis ---------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for DPS analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTDPSANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// TestDPSAnalysis pass
//===----------------------------------------------------------------------===//
class TestDPSAnalysis
    : public mlir::aster::test::impl::TestDPSAnalysisBase<TestDPSAnalysis> {
public:
  using TestDPSAnalysisBase::TestDPSAnalysisBase;

  void runOnOperation() override {
    getOperation()->walk([&](FunctionOpInterface op) -> void {
      SSAMap ssaMap;
      ssaMap.populateMap(op);
      FailureOr<DPSAnalysis> analysis = DPSAnalysis::create(op);
      if (failed(analysis)) {
        op->emitError() << "failed to run DPS analysis";
        return signalPassFailure();
      }
      raw_prefixed_ostream os(llvm::outs(), "// ");
      os << "function: " << op.getNameAttr() << "\n";
      ssaMap.printMapMembers(os);
      os << "\n";
      analysis->print(os, ssaMap);
      os << "\n";
    });
  }
};
} // namespace
