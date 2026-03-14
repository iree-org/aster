//===- SCFRotate.cpp - Rotate loop body around insertion point --------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass moves ops marked with sched.rotate_head -- together with their
// transitive in-block dependencies -- to the front of scf.for loop bodies.
//
// The "most dominating insertion point" is the earliest transitive dependency
// of any rotate_head op. After rotation, that op becomes the first op in the
// loop body, followed by the rest of the dependency subgraph and the
// rotate_head ops themselves, all in their original relative order.
//
// Designed to run after pipelining.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
#define GEN_PASS_DEF_SCFROTATE
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

namespace mlir::aster {
namespace {

constexpr StringLiteral kSchedRotateHeadAttr = "sched.rotate_head";

/// Collect the transitive in-block dependency closure of rotate_head ops.
/// Every in-block op that a rotate_head op transitively depends on is
/// included in the move set. The earliest such op -- the "most dominating
/// insertion point" -- becomes the first op in the loop body after rotation.
/// Returns the rotate_head ops in program order.
static SmallVector<Operation *> collectMoveSet(Block *block,
                                               DenseSet<Operation *> &moveSet) {
  SmallVector<Operation *> rotateOps;
  for (Operation &op : *block) {
    if (op.hasAttr(kSchedRotateHeadAttr)) {
      rotateOps.push_back(&op);
      moveSet.insert(&op);
    }
  }

  // Transitively collect all in-block dependencies.
  SmallVector<Operation *> worklist(rotateOps.begin(), rotateOps.end());
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != block)
        continue;
      if (moveSet.insert(defOp).second)
        worklist.push_back(defOp);
    }
  }

  return rotateOps;
}

/// Move all ops in the move set to the block front, preserving their
/// relative order (which respects SSA dominance by construction).
static void moveToBlockFront(Block *block,
                             const DenseSet<Operation *> &moveSet) {
  Operation *cursor = nullptr;
  for (Operation &op : llvm::make_early_inc_range(*block)) {
    if (!moveSet.contains(&op))
      continue;
    if (cursor)
      op.moveAfter(cursor);
    else
      op.moveBefore(&block->front());
    cursor = &op;
  }
}

struct SCFRotatePass : public impl::SCFRotateBase<SCFRotatePass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {
      Block *block = forOp.getBody();

      DenseSet<Operation *> moveSet;
      SmallVector<Operation *> rotateOps = collectMoveSet(block, moveSet);
      if (rotateOps.empty())
        return;

      moveToBlockFront(block, moveSet);

      // Strip rotate_head attributes.
      for (Operation *op : rotateOps)
        op->removeAttr(kSchedRotateHeadAttr);
    });
  }
};

} // namespace
} // namespace mlir::aster
