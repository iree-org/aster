// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/NormalForm/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

#define GEN_PASS_DEF_LOWERWATERNORMALFORMMODULEPASS
#include "water/Dialect/NormalForm/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// LowerWaterNormalFormModulePattern
//===----------------------------------------------------------------------===//

/// Lower `water_normalform.module` to `builtin.module`, discarding normal form
/// attributes.
class LowerWaterNormalFormModulePattern
    : public OpRewritePattern<water_normalform::ModuleOp> {
public:
  using OpRewritePattern<water_normalform::ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(water_normalform::ModuleOp nfModule,
                                PatternRewriter &rewriter) const override {
    // Check if parent is a builtin module - if so, inline contents into parent.
    if (auto parentModule = dyn_cast<ModuleOp>(nfModule->getParentOp())) {
      rewriter.setInsertionPoint(nfModule);
      for (Operation &op : llvm::make_early_inc_range(*nfModule.getBody()))
        rewriter.moveOpBefore(&op, nfModule);
      rewriter.eraseOp(nfModule);
      return success();
    }

    // Otherwise, create a new builtin module.
    ModuleOp builtinModule =
        ModuleOp::create(rewriter, nfModule.getLoc(), nfModule.getName());

    // Move all blocks from the water_normalform module to the builtin module.
    rewriter.inlineRegionBefore(nfModule.getRegion(), builtinModule.getBody());

    // Remove the empty terminator block that was automatically added by the
    // builder.
    rewriter.eraseBlock(&builtinModule.getBodyRegion().back());

    rewriter.eraseOp(nfModule);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerWaterNormalFormModulePass
//===----------------------------------------------------------------------===//

struct LowerWaterNormalFormModulePass
    : public ::impl::LowerWaterNormalFormModulePassBase<
          LowerWaterNormalFormModulePass> {
  using LowerWaterNormalFormModulePassBase::LowerWaterNormalFormModulePassBase;

  void runOnOperation() override {
    Operation *root = getOperation();

    if (auto rootModule = dyn_cast<ModuleOp>(root)) {
      int64_t count = llvm::count_if(rootModule.getBody()->getOperations(),
                                     llvm::IsaPred<water_normalform::ModuleOp>);

      if (count > 1) {
        rootModule.emitError() << "expected at most one top-level "
                               << water_normalform::ModuleOp::getOperationName()
                               << ", found " << count;
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerWaterNormalFormModulePattern>(&getContext());

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> water_normalform::createLowerWaterNormalFormModulePass() {
  return std::make_unique<LowerWaterNormalFormModulePass>();
}
