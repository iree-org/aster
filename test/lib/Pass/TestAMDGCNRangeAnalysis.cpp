//===- TestAMDGCNRangeAnalysis.cpp - Test Range Constraint Analysis -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for register range constraint analysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/IR/SSAMap.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTAMDGCNRANGEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestAMDGCNRangeAnalysis pass
//===----------------------------------------------------------------------===//
struct TestAMDGCNRangeAnalysis
    : public mlir::aster::test::impl::TestAMDGCNRangeAnalysisBase<
          TestAMDGCNRangeAnalysis> {
  using TestAMDGCNRangeAnalysisBase::TestAMDGCNRangeAnalysisBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Create the SSA map and prefixed output stream.
    SSAMap ssaMap;
    ssaMap.populateMap(op);
    raw_prefixed_ostream os(llvm::outs(), "// ");
    os << "=== Range Constraint Analysis ===\n";
    os << "SSA map:\n";
    ssaMap.printMapMembers(os);
    os << "\n";

    // Walk through kernels and run analysis on each one.
    op->walk([&](FunctionOpInterface kernel) {
      os << "Symbol: " << kernel.getName() << "\n";

      // Create the range constraint analysis.
      FailureOr<RangeConstraintAnalysis> analysis =
          RangeConstraintAnalysis::create(kernel);
      if (failed(analysis)) {
        kernel.emitError() << "Failed to run range constraint analysis";
        return signalPassFailure();
      }

      // Print the range constraints.
      ArrayRef<RangeConstraint> ranges = analysis->getRanges();
      if (ranges.empty()) {
        os << "No range constraints\n\n";
        return;
      }
      os << "Range constraints:\n";
      os.indent();
      for (auto [i, range] : llvm::enumerate(ranges)) {
        os << "Constraint " << i << ": ";
        range.print(os, ssaMap);
        os << "\n";
      }
      os.unindent();
    });
  }
};
} // namespace
