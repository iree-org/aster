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
  StringRef schedPrefix = "sched.";
  op->walk<WalkOrder::PostOrder>([&](CallOpInterface callOp) {
    if (callOp.getOperation()->getParentOfType<ExecuteRegionOp>())
      return WalkResult::advance();

    // Skip wrapping calls with sched.* attrs -- they won't be inlined
    // (profitability rejects them) and wrapping causes attr loss.
    bool hasSched = false;
    for (NamedAttribute attr : callOp->getAttrs()) {
      if (attr.getName().strref().starts_with(schedPrefix)) {
        hasSched = true;
        break;
      }
    }
    if (hasSched)
      return WalkResult::advance();

    rewriter.setInsertionPoint(callOp);
    Location loc = callOp.getLoc();

    auto executeRegionOp =
        ExecuteRegionOp::create(rewriter, loc, callOp->getResultTypes());

    rewriter.replaceAllOpUsesWith(callOp, executeRegionOp.getResults());

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
