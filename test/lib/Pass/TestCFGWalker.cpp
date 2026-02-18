//===- TestCFGWalker.cpp - Test CFG Walker
//---------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass that walks the control flow graph and
// prints each IR element (operations and control flow edges) encountered.
//
//===----------------------------------------------------------------------===//

#include "aster/IR/CFG.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/Support/PrefixedOstream.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTCFGWALKER
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// Print CFG Walker
//===----------------------------------------------------------------------===//
class PrintCFGWalker : public CFGWalker<PrintCFGWalker> {
public:
  explicit PrintCFGWalker(llvm::raw_ostream &os) : os(os) {}

  LogicalResult visitOp(Operation *op) {
    os << "    " << OpWithFlags(op, OpPrintingFlags().skipRegions()) << "\n";
    return success();
  }

  LogicalResult visitControlFlowEdge(const BranchPoint &branchPoint,
                                     const Successor &successor) {
    os << "cfg: ";
    if (branchPoint.isEntryPoint())
      os << "entry";
    else
      os << OpWithFlags(branchPoint.getPoint(),
                        OpPrintingFlags().skipRegions());
    os << " -> ";
    if (successor.isBlock())
      os << BlockWithFlags(successor.getTarget<Block *>(),
                           BlockWithFlags::PrintMode::PrintAsQualifiedOperand,
                           OpPrintingFlags().skipRegions());
    else
      os << OpWithFlags(successor.getTarget<Operation *>(),
                        OpPrintingFlags().skipRegions());
    os << "\n";
    return success();
  }

private:
  llvm::raw_ostream &os;
};

//===----------------------------------------------------------------------===//
// TestCFGWalker pass
//===----------------------------------------------------------------------===//
class TestCFGWalker
    : public mlir::aster::test::impl::TestCFGWalkerBase<TestCFGWalker> {
public:
  using TestCFGWalkerBase::TestCFGWalkerBase;

  void runOnOperation() override {
    raw_prefixed_ostream os(llvm::outs(), "// ");
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      os << "function: " << funcOp.getNameAttr() << "\n";

      PrintCFGWalker walker(os);
      if (failed(walker.walk(funcOp)))
        signalPassFailure();
      os << "\n";
    });
  }
};
} // namespace
