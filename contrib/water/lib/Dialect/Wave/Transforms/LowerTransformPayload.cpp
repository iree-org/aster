// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace wave {
#define GEN_PASS_DEF_WATERWAVELOWERTRANSFORMPAYLOADPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

/// Lower `transform.payload` to `builtin.module`, discarding the normal form
/// attributes. The Detect pass nests payload bodies inside an extra
/// `builtin.module` to satisfy the `SymbolTable` requirement of symbols (e.g.
/// `func.func`). When such a single-module body is encountered, this pattern
/// inlines the inner module's contents directly to avoid leaving a redundant
/// `builtin.module` wrapper behind.
class LowerTransformPayloadPattern
    : public OpRewritePattern<transform::PayloadOp> {
public:
  using OpRewritePattern<transform::PayloadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(transform::PayloadOp payload,
                                PatternRewriter &rewriter) const override {
    Block &body = payload.getBody().front();
    Operation *parent = payload->getParentOp();

    // If the body consists of a single inner `builtin.module`, treat that
    // module's body as the effective payload contents.
    ModuleOp innerModule;
    if (llvm::hasSingleElement(body))
      innerModule = dyn_cast<ModuleOp>(&body.front());
    Block *contents = innerModule ? innerModule.getBody() : &body;

    // If the parent is a builtin module, inline the effective contents into
    // it, replacing the payload.
    if (parent && isa<ModuleOp>(parent)) {
      rewriter.setInsertionPoint(payload);
      for (Operation &op : llvm::make_early_inc_range(*contents))
        rewriter.moveOpBefore(&op, payload);
      rewriter.eraseOp(payload);
      return success();
    }

    // Otherwise create a fresh `builtin.module` that takes the place of the
    // payload and holds the effective contents.
    rewriter.setInsertionPoint(payload);
    ModuleOp newModule = ModuleOp::create(rewriter, payload.getLoc());
    Block *newBlock = newModule.getBody();
    for (Operation &op : llvm::make_early_inc_range(*contents))
      rewriter.moveOpBefore(&op, newBlock, newBlock->end());
    rewriter.eraseOp(payload);
    return success();
  }
};

struct WaterWaveLowerTransformPayloadPass
    : public wave::impl::WaterWaveLowerTransformPayloadPassBase<
          WaterWaveLowerTransformPayloadPass> {
  using WaterWaveLowerTransformPayloadPassBase::
      WaterWaveLowerTransformPayloadPassBase;

  void runOnOperation() override {
    Operation *root = getOperation();

    if (auto rootModule = dyn_cast<ModuleOp>(root)) {
      int64_t count = llvm::count_if(rootModule.getBody()->getOperations(),
                                     llvm::IsaPred<transform::PayloadOp>);

      if (count > 1) {
        rootModule.emitError()
            << "expected at most one top-level "
            << transform::PayloadOp::getOperationName() << ", found " << count;
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTransformPayloadPattern>(&getContext());

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
