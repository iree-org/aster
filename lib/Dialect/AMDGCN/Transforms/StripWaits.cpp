// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNSTRIPWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct AMDGCNStripWaits
    : public amdgcn::impl::AMDGCNStripWaitsBase<AMDGCNStripWaits> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void AMDGCNStripWaits::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());

  // Erase every wait op (CDNA and gfx1250 families).
  SmallVector<Operation *> toErase;
  op->walk([&](Operation *innerOp) {
    if (isa<WaitOp, WaitGfx1250Op, SWaitcnt, SWaitLoadcnt, SWaitStorecnt,
            SWaitDscnt, SWaitKmcnt, SWaitTensorcnt, SWaitLoadcntDscnt,
            SWaitStorecntDscnt>(innerOp))
      toErase.push_back(innerOp);
  });
  for (Operation *waitOp : toErase)
    rewriter.eraseOp(waitOp);
}
