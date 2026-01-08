//===- RemoveTestInst.cpp -------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REMOVETESTINST
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct RemoveTestInst
    : public mlir::aster::amdgcn::impl::RemoveTestInstBase<RemoveTestInst> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void RemoveTestInst::runOnOperation() {
  Operation *op = getOperation();
  SmallVector<amdgcn::TestInstOp> toErase;
  op->walk([&](amdgcn::TestInstOp testOp) { toErase.push_back(testOp); });
  for (auto testOp : toErase)
    testOp.erase();
}

