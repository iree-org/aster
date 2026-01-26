//===- InsertWaits.cpp - Insert wait operations ---------------------------===//
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
#define GEN_PASS_DEF_AMDGCNINSERTWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct AMDGCNInsertWaits
    : public mlir::aster::amdgcn::impl::AMDGCNInsertWaitsBase<
          AMDGCNInsertWaits> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void AMDGCNInsertWaits::runOnOperation() {
  Operation *op = getOperation();

  // Collect all load ops.
  SmallVector<amdgcn::LoadOp> ops;
  op->walk([&](amdgcn::LoadOp op) { ops.push_back(op); });

  OpBuilder builder(op->getContext());
  // Insert waits before each use.
  for (amdgcn::LoadOp loadOp : ops) {
    for (Operation *userOp : loadOp.getResult().getUsers()) {
      builder.setInsertionPoint(userOp);
      builder.create<amdgcn::WaitOp>(userOp->getLoc(), loadOp.getToken());
    }
  }
}
