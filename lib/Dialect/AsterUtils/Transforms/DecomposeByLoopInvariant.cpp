//===- DecomposeByLoopInvariant.cpp - Split loop-invariant operands -------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "decompose-by-loop-invariant"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_DECOMPOSEBYLOOPINVARIANT
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
struct DecomposeByLoopInvariant
    : public aster_utils::impl::DecomposeByLoopInvariantBase<
          DecomposeByLoopInvariant> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// LICM operand splitting
//===----------------------------------------------------------------------===//

/// Split the loop-invariant operands of `op` (an AddiOp or MuliOp) into a
/// separate op of the same kind, insert a passthrough barrier on its result,
/// then combine the barrier with the remaining variant operands in a new outer
/// op that replaces `op`.
///
/// The transformation requires at least two invariant operands (so that the
/// inner op is valid) and at least one variant operand (otherwise the whole op
/// is already hoistable and nothing needs to be done).
template <typename OpTy>
static void splitLICMOperands(IRRewriter &rewriter, OpTy op) {
  LDBG() << "Splitting LICM operands of: " << op;
  auto loop = op->template getParentOfType<LoopLikeOpInterface>();
  if (!loop) {
    LDBG() << " - skipping (not inside a loop)";
    return;
  }

  SmallVector<Value> invariant, variant;
  for (Value v : op.getInputs()) {
    Operation *definingOp = v.getDefiningOp();
    if (loop.isDefinedOutsideOfLoop(v) ||
        (definingOp && definingOp->hasTrait<OpTrait::ConstantLike>())) {
      invariant.push_back(v);
      continue;
    }
    variant.push_back(v);
  }

  // Need at least two invariant operands to form a valid inner op (addi/muli
  // require at least 2 inputs), and at least one variant operand so that the
  // original op is not already fully hoistable.
  if (invariant.size() < 2 || variant.empty()) {
    LDBG() << " - skipping (invariant=" << invariant.size()
           << ", variant=" << variant.size() << "): " << op;
    return;
  }

  LDBG() << " - decomposing (invariant=" << invariant.size()
         << ", variant=" << variant.size() << "): " << op;

  Location loc = op.getLoc();
  Type indexType = rewriter.getIndexType();
  StringAttr tag = rewriter.getStringAttr("__decompose_ops__");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Inner op: combines all loop-invariant operands.
  Value invariantResult = OpTy::create(rewriter, loc, indexType, invariant);
  LDBG() << "  - inner op: " << invariantResult.getDefiningOp();

  // Passthrough barrier: prevents the optimizer from re-merging the two ops.
  Value barrier = PassthroughOp::create(rewriter, loc, invariantResult, tag);
  LDBG() << "  - barrier: " << barrier.getDefiningOp();

  // Outer op: combines the barrier result with all variant operands.
  variant.push_back(barrier);
  rewriter.replaceOpWithNewOp<OpTy>(op, indexType, variant);
  LDBG() << "  - outer op inserted";
}

//===----------------------------------------------------------------------===//
// DecomposeByLoopInvariant pass
//===----------------------------------------------------------------------===//

void DecomposeByLoopInvariant::runOnOperation() {
  IRRewriter rewriter(&getContext());
  getOperation()->walk([&](Operation *op) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (auto addi = dyn_cast<AddiOp>(op))
      splitLICMOperands(rewriter, addi);
    else if (auto muli = dyn_cast<MuliOp>(op))
      splitLICMOperands(rewriter, muli);
  });
}
