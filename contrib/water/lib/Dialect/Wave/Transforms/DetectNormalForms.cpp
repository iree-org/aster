// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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
static SmallVector<transform::NormalFormAttrInterface>
collectWaveWaterNormalForms(MLIRContext *ctx) {
  SmallVector<transform::NormalFormAttrInterface> normalForms;
  for (unsigned bit = 0, lastBit = WaveWaterNormalFormAttr::getLastSetBit();
       bit <= lastBit; ++bit) {
    WaveWaterNormalForm form =
        static_cast<WaveWaterNormalForm>(static_cast<uint32_t>(1) << bit);
    normalForms.push_back(WaveWaterNormalFormAttr::get(ctx, form));
  }
  return normalForms;
}

/// Check whether the payload satisfies the given normal form without emitting
/// any diagnostics.
static bool silentlySatisfies(transform::PayloadOp payload,
                              transform::NormalFormAttrInterface normalForm) {
  ScopedDiagnosticHandler handler(payload->getContext(),
                                  [](Diagnostic &) { return success(); });
  DiagnosedSilenceableFailure result =
      normalForm.checkOperation(payload.getOperation());
  bool ok = result.succeeded();
  // Discard any silenceable diagnostics so they are not reported elsewhere.
  (void)result.silence();
  return ok;
}

/// Add to `payload` all of the candidate normal forms that it satisfies. The
/// `WaveWaterNormalFormAttr` bits are combined into a single attribute so the
/// resulting array does not contain duplicate normal-form attribute kinds (as
/// required by the `transform.payload` verifier).
/// Returns true if any change was made.
static bool
inferWaterNormalForms(transform::PayloadOp payload,
                      ArrayRef<transform::NormalFormAttrInterface> candidates) {
  MLIRContext *ctx = payload.getContext();

  // Separate non-wave normal forms (kept as-is) from wave normal forms that
  // need to be merged into a single attribute.
  WaveWaterNormalForm currentWave = WaveWaterNormalForm::None;
  SmallVector<Attribute> nonWaveForms;
  for (Attribute attr : payload.getNormalForms()) {
    if (auto waveForm = llvm::dyn_cast<WaveWaterNormalFormAttr>(attr)) {
      currentWave = currentWave | waveForm.getValue();
      continue;
    }
    nonWaveForms.push_back(attr);
  }

  WaveWaterNormalForm finalWave = currentWave;
  bool changed = false;
  for (transform::NormalFormAttrInterface candidate : candidates) {
    auto candidateWave = llvm::dyn_cast<WaveWaterNormalFormAttr>(candidate);
    if (!candidateWave)
      continue;
    if (bitEnumContainsAll(finalWave, candidateWave.getValue()))
      continue;
    if (silentlySatisfies(payload, candidate)) {
      finalWave = finalWave | candidateWave.getValue();
      changed = true;
    }
  }

  if (!changed)
    return false;

  SmallVector<Attribute> newForms = std::move(nonWaveForms);
  if (finalWave != WaveWaterNormalForm::None)
    newForms.push_back(WaveWaterNormalFormAttr::get(ctx, finalWave));
  payload.setNormalFormsAttr(ArrayAttr::get(ctx, newForms));
  return true;
}

namespace {

//===----------------------------------------------------------------------===//
// DetectWaterNormalFormsPattern
//===----------------------------------------------------------------------===//

/// Wrap a builtin module inside a `transform.payload` op (which becomes the
/// new container at the module's previous location) and infer which Wave
/// dialect normal forms are satisfied. The wrapped module is preserved as a
/// child of the payload so symbols (e.g. `func.func`) still have a
/// `SymbolTable` parent.
class DetectWaterNormalFormsPattern : public OpRewritePattern<ModuleOp> {
public:
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp module,
                                PatternRewriter &rewriter) const override {
    // Skip modules that are already wrapped by a transform.payload, so the
    // walk does not endlessly re-wrap the same payload's body.
    if (isa_and_nonnull<transform::PayloadOp>(module->getParentOp()))
      return failure();

    MLIRContext *ctx = module.getContext();
    Location loc = module.getLoc();

    auto emptyArray = ArrayAttr::get(ctx, {});
    rewriter.setInsertionPoint(module);
    transform::PayloadOp payload =
        rewriter.create<transform::PayloadOp>(loc, emptyArray);
    Block &payloadBody = payload.getBody().emplaceBlock();
    rewriter.moveOpBefore(module, &payloadBody, payloadBody.end());

    inferWaterNormalForms(payload, collectWaveWaterNormalForms(ctx));
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

    // Update existing transform.payload ops in case new normal forms apply.
    rootOp->walk([&](transform::PayloadOp payload) {
      inferWaterNormalForms(payload, collectWaveWaterNormalForms(ctx));
    });

    // Run the pattern rewriter on any nested builtin modules.
    RewritePatternSet patterns(&getContext());
    patterns.add<DetectWaterNormalFormsPattern>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // The pattern cannot wrap the root operation (we cannot replace the root
    // op of a pass). When the root is a `builtin.module`, manually wrap its
    // body so that its contents live inside a `transform.payload` containing
    // a fresh inner `builtin.module` (which carries the `SymbolTable` trait
    // required by symbols such as `func.func`).
    auto rootModule = dyn_cast<ModuleOp>(rootOp);
    if (!rootModule)
      return;

    // Skip if the root module is already exactly a payload-wrapped module
    // (i.e. its only child is a transform.payload).
    auto rootOps = rootModule.getBodyRegion().getOps();
    if (!rootOps.empty() &&
        llvm::all_of(rootOps, llvm::IsaPred<transform::PayloadOp>))
      return;

    OpBuilder builder(ctx);
    Location loc = rootModule.getLoc();
    auto emptyArray = ArrayAttr::get(ctx, {});

    // Create the new container payload and an inner builtin.module that will
    // host the original root module's contents.
    transform::PayloadOp payload =
        transform::PayloadOp::create(builder, loc, emptyArray);
    Block &payloadBody = payload.getBody().emplaceBlock();

    builder.setInsertionPointToStart(&payloadBody);
    ModuleOp innerModule = ModuleOp::create(builder, loc);

    // Splice the original root module's operations into the inner module.
    Block *rootBlock = rootModule.getBody();
    Block *innerBlock = innerModule.getBody();
    innerBlock->getOperations().splice(innerBlock->getOperations().end(),
                                       rootBlock->getOperations());

    // Place the new payload as the only child of the root module.
    rootBlock->push_back(payload);
    inferWaterNormalForms(payload, collectWaveWaterNormalForms(ctx));
  }
};

} // end anonymous namespace
