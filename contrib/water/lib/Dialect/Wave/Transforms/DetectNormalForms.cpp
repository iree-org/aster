// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace wave;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEDETECTWATERNORMALFORMSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Collect all Wave dialect normal forms that can be inferred.
static SmallVector<water_normalform::WaterNormalFormAttrInterface>
collectWaveWaterNormalForms(MLIRContext *ctx) {
  SmallVector<water_normalform::WaterNormalFormAttrInterface> normalForms;
  for (unsigned bit = 0, lastBit = WaveWaterNormalFormAttr::getLastSetBit();
       bit <= lastBit; ++bit) {
    WaveWaterNormalForm form =
        static_cast<WaveWaterNormalForm>(static_cast<uint32_t>(1) << bit);
    normalForms.push_back(WaveWaterNormalFormAttr::get(ctx, form));
  }
  return normalForms;
}

namespace {

//===----------------------------------------------------------------------===//
// DetectWaterNormalFormsPattern
//===----------------------------------------------------------------------===//

/// Wrap a builtin module in a water_normalform module and infer which Wave
/// normal forms apply.
class DetectWaterNormalFormsPattern : public OpRewritePattern<ModuleOp> {
public:
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp module,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = module.getContext();
    Location loc = module.getLoc();

    water_normalform::ModuleOp nfModule =
        water_normalform::ModuleOp::create(loc, {}, module.getName());

    Block &nfBody = nfModule.getBodyRegion().front();
    Block &moduleBody = module.getBodyRegion().front();
    rewriter.inlineBlockBefore(&moduleBody, &nfBody, nfBody.end());

    nfModule.inferWaterNormalForms(collectWaveWaterNormalForms(ctx));

    rewriter.setInsertionPoint(module);
    rewriter.insert(nfModule);
    rewriter.eraseOp(module);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaterWaveDetectWaterNormalFormsPass
//===----------------------------------------------------------------------===//

struct WaterWaveDetectWaterNormalFormsPass
    : public wave::impl::WaterWaveDetectWaterNormalFormsPassBase<
          WaterWaveDetectWaterNormalFormsPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *ctx = rootOp->getContext();

    // update exisiting water_normalform Modules
    rootOp->walk([&](water_normalform::ModuleOp nfModule) {
      nfModule.inferWaterNormalForms(collectWaveWaterNormalForms(ctx));
    });

    // Run the pattern rewriter on any nested builtin modules.
    RewritePatternSet patterns(&getContext());
    patterns.add<DetectWaterNormalFormsPattern>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // If the root operation is a ModuleOp, wrap its contents in a
    // water_normalform module. Only the root module can be a builtin module at
    // this point. If all contained operations are already water_normalform
    // module operations, we don't need to wrap the root modules body in another
    // water_normalform.module.
    if (auto rootModule = dyn_cast<ModuleOp>(rootOp)) {
      auto ops = rootModule.getBodyRegion().getOps();
      if (!ops.empty() &&
          llvm::all_of(ops, llvm::IsaPred<water_normalform::ModuleOp>))
        return;

      water_normalform::ModuleOp nfModule =
          water_normalform::ModuleOp::create(rootModule.getLoc(), {});

      Block *currentBlock = rootModule.getBody();
      Block *newBlock = nfModule.getBody();

      newBlock->getOperations().splice(newBlock->getOperations().end(),
                                       currentBlock->getOperations());

      rootModule.getBody()->push_back(nfModule);
      nfModule.inferWaterNormalForms(collectWaveWaterNormalForms(ctx));
    }
  }
};

} // end anonymous namespace
