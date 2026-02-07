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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

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

    // Walk through kernels and run analysis on each one.
    op->walk([&](FunctionOpInterface kernel) {
      llvm::errs() << "// Function: " << kernel.getName() << "\n";

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
        llvm::errs() << "// No range constraints\n\n";
        return;
      }
      llvm::errs() << "// Range constraints:\n";
      for (auto [i, range] : llvm::enumerate(ranges)) {
        llvm::errs() << "//  ";
        range.print(llvm::errs());
        llvm::errs() << "\n";
      }
      llvm::errs() << "\n";
    });
  }
};
} // namespace
