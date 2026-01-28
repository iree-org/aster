//===- TestLDSInterferenceGraph.cpp - Test LDS Interference Graph --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass for the LDS interference graph.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/LDSInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTLDSINTERFERENCEGRAPH
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestLDSInterferenceGraph pass
//===----------------------------------------------------------------------===//
struct TestLDSInterferenceGraph
    : public mlir::aster::test::impl::TestLDSInterferenceGraphBase<
          TestLDSInterferenceGraph> {
  using TestLDSInterferenceGraphBase::TestLDSInterferenceGraphBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    FailureOr<LDSInterferenceGraph> graph =
        LDSInterferenceGraph::create(op, domInfo);
    if (failed(graph))
      return signalPassFailure();

    graph->print(llvm::errs());
  }
};
} // namespace
