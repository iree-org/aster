//===- InlineExecuteRegion.cpp - Inline execute_region operations --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/Transforms/Passes.h"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_INLINEEXECUTEREGION
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// InlineExecuteRegion pass
//===----------------------------------------------------------------------===//
struct InlineExecuteRegion
    : public aster_utils::impl::InlineExecuteRegionBase<InlineExecuteRegion> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void InlineExecuteRegion::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());
  op->walk<WalkOrder::PostOrder>([&](ExecuteRegionOp executeRegionOp) {
    rewriter.setInsertionPoint(executeRegionOp);
    Block *block = executeRegionOp.getBody();
    auto yieldOp = cast<YieldOp>(block->getTerminator());
    rewriter.inlineBlockBefore(block, executeRegionOp);
    rewriter.replaceOp(executeRegionOp, yieldOp.getResults());
    rewriter.eraseOp(yieldOp);
    return WalkResult::skip();
  });
}
