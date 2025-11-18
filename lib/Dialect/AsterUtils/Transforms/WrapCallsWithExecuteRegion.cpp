//===- WrapCallsWithExecuteRegion.cpp - Wrap calls with execute_region ---===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/Transforms/Passes.h"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_WRAPCALLSWITHEXECUTEREGION
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// WrapCallsWithExecuteRegion pass
//===----------------------------------------------------------------------===//
struct WrapCallsWithExecuteRegion
    : public aster_utils::impl::WrapCallsWithExecuteRegionBase<
          WrapCallsWithExecuteRegion> {
public:
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void WrapCallsWithExecuteRegion::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());
  op->walk<WalkOrder::PostOrder>([&](CallOpInterface callOp) {
    if (callOp.getOperation()->getParentOfType<ExecuteRegionOp>())
      return WalkResult::advance();
    rewriter.setInsertionPoint(callOp);
    Location loc = callOp.getLoc();

    // Create the execute_region operation.
    auto executeRegionOp =
        ExecuteRegionOp::create(rewriter, loc, callOp->getResultTypes());

    // Replace the call uses with the execute_region results.
    rewriter.replaceAllOpUsesWith(callOp, executeRegionOp.getResults());

    // Move the call into the execute_region body.
    Block *block = rewriter.createBlock(&executeRegionOp.getRegion());
    rewriter.moveOpBefore(callOp, block, block->end());
    YieldOp::create(rewriter, loc, callOp->getResults());
    return WalkResult::skip();
  });
}
