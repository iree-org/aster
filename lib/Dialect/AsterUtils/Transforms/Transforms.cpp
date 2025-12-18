//===- Transforms.cpp - Common AsterUtils transforms ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/Transforms/Transforms.h"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

void aster_utils::wrapCallsWithExecuteRegion(Operation *op) {
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

void aster_utils::inlineExecuteRegions(Operation *op) {
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
