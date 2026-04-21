// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Pipelines.h"

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

using namespace mlir;

void mlir::water::registerWaterPipelines() {
  PassPipelineRegistration<>(
      "water-middle-end-lowering",
      "Lower Wave dialect through normal forms to upstream MLIR dialects.",
      [](OpPassManager &pm) {
        // Wave passes anchor on `builtin.module` because `transform.payload`
        // is not `IsolatedFromAbove`. They walk the IR for `transform.payload`
        // ops nested inside their root module, so the passes can be added to
        // the top-level pass manager directly.
        pm.addPass(wave::createWaterWaveDetectWaterNormalFormsPass());
        pm.addPass(wave::createWaterWavePropagateElementsPerThreadPass());
        pm.addPass(wave::createWaterWaveResolveDistributedAllocationsPass());
        pm.addPass(wave::createWaterWaveDetectWaterNormalFormsPass());
        pm.addPass(wave::createLowerWaveToMLIRPass());
        pm.addPass(wave::createWaterWaveLowerTransformPayloadPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}
