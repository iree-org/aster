//===- Pipelines.cpp - CodeGen Pass Pipelines -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Passes.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::aster;

static void buildCodeGenPassPipeline(OpPassManager &pm) {
  pm.addPass(createLegalizer());
  pm.addPass(createToIntArith());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizePtrOps());
  pm.addPass(aster_utils::createOptimizePtrAdd());
  pm.addPass(createCodeGen());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlir::aster::registerCodeGenPassPipeline() {
  PassPipelineRegistration<>("aster-codegen-pipeline",
                             "Run the aster code generation pipeline",
                             buildCodeGenPassPipeline);
}
