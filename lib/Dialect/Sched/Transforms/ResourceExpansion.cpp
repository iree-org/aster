//===- ResourceExpansion.cpp - Expand sched.loop_resource -------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass moves sched.loop_resource ops out of their enclosing loops and
// expands each region into the appropriate location around the loop.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Sched/IR/SchedOps.h"
#include "aster/Dialect/Sched/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster::sched {
#define GEN_PASS_DEF_RESOURCEEXPANSION
#include "aster/Dialect/Sched/Transforms/Passes.h.inc"
} // namespace mlir::aster::sched

using namespace mlir;
using namespace mlir::aster::sched;

namespace {

/// Inline the body of a region (excluding the terminator) at the current
/// builder insertion point, using `mapping` to remap values. Returns the
/// operands of the terminating sched.yield (mapped to the new scope).
static SmallVector<Value> inlineRegionBody(Region &region, OpBuilder &builder,
                                           IRMapping &mapping) {
  Block &block = region.front();
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);
  YieldOp yield = cast<YieldOp>(block.getTerminator());
  SmallVector<Value> results;
  for (Value v : yield.getValues())
    results.push_back(mapping.lookupOrDefault(v));
  return results;
}

struct ResourceExpansionPass
    : public mlir::aster::sched::impl::ResourceExpansionBase<
          ResourceExpansionPass> {
  using ResourceExpansionBase::ResourceExpansionBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    SmallVector<LoopResourceOp> resources;
    root->walk([&](LoopResourceOp op) { resources.push_back(op); });

    for (LoopResourceOp resourceOp : resources)
      if (failed(expandResource(resourceOp)))
        return signalPassFailure();
  }

  LogicalResult expandResource(LoopResourceOp resourceOp) {
    auto loopOp = resourceOp->getParentOfType<LoopLikeOpInterface>();
    if (!loopOp)
      return resourceOp.emitOpError("not inside a loop-like op");

    if (numIts > 0)
      return resourceOp.emitOpError(
          "numIts > 0 rotation not yet implemented; use numIts = 0");

    Operation *loopBase = loopOp.getOperation();
    Location loc = resourceOp.getLoc();
    OpBuilder builder(loopBase);

    // Inline the allocate region before the loop to get the resource handles.
    // The allocate region has no block args.
    IRMapping allocMapping;
    SmallVector<Value> allocValues =
        inlineRegionBody(resourceOp.getAllocate(), builder, allocMapping);

    // Inline the forward region before the loop to get initial iter_arg values.
    // The forward region's block args correspond to the allocate yields.
    IRMapping fwdMapping;
    Block &fwdBlock = resourceOp.getForward().front();
    for (auto [arg, val] : llvm::zip(fwdBlock.getArguments(), allocValues))
      fwdMapping.map(arg, val);
    SmallVector<Value> initialFwd =
        inlineRegionBody(resourceOp.getForward(), builder, fwdMapping);

    // Inline the deallocate region after the loop (if present).
    if (!resourceOp.getDeallocate().empty()) {
      builder.setInsertionPointAfter(loopBase);
      IRMapping deallMapping;
      Block &deallBlock = resourceOp.getDeallocate().front();
      for (auto [arg, val] : llvm::zip(deallBlock.getArguments(), allocValues))
        deallMapping.map(arg, val);
      inlineRegionBody(resourceOp.getDeallocate(), builder, deallMapping);
    }

    // Inline the fence region at the end of the loop body (if present).
    if (!resourceOp.getFence().empty()) {
      Block *loopBody = &loopBase->getRegion(0).front();
      builder.setInsertionPoint(loopBody->getTerminator());
      IRMapping fenceMapping;
      Block &fenceBlock = resourceOp.getFence().front();
      for (auto [arg, val] : llvm::zip(fenceBlock.getArguments(), allocValues))
        fenceMapping.map(arg, val);
      inlineRegionBody(resourceOp.getFence(), builder, fenceMapping);
    }

    // Replace uses of the resource op's results with the forward values.
    for (auto [res, fwd] : llvm::zip(resourceOp.getResults(), initialFwd))
      res.replaceAllUsesWith(fwd);

    resourceOp.erase();
    return success();
  }
};

} // namespace
