//===- HoistOps.cpp -------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_HOISTOPS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// HoistOps pass
//===----------------------------------------------------------------------===//
struct HoistOps : public amdgcn::impl::HoistOpsBase<HoistOps> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// HoistOps pass
//===----------------------------------------------------------------------===//

void HoistOps::runOnOperation() {
  getOperation()->walk([&](FunctionOpInterface op) {
    // Get entry block.
    Block *entryBlock{};
    if (Region &region = op.getFunctionBody(); !region.empty())
      entryBlock = &region.front();

    // Early exit if no entry block.
    if (!entryBlock || entryBlock->empty())
      return;

    // Collect all ops to hoist.
    SetVector<Operation *> opList;
    op.walk([&](Operation *op) {
      if (!isa<AllocaOp, ThreadIdOp, BlockDimOp, BlockIdOp, LoadArgOp>(op) &&
          !op->hasTrait<OpTrait::ConstantLike>())
        return;
      opList.insert(op);
    });
    SetVector<Operation *> sortedOps = mlir::topologicalSort(opList);
    IRRewriter rewriter(op.getContext());
    for (Operation *hoistOp : llvm::reverse(sortedOps.getArrayRef()))
      rewriter.moveOpBefore(hoistOp, entryBlock, entryBlock->begin());

    auto &dominance = getAnalysis<DominanceInfo>();
    eliminateCommonSubExpressions(rewriter, dominance, op, nullptr);
  });
}
