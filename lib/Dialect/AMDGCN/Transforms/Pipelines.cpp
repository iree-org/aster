//===- Pipelines.cpp - AMDGCN Pass Pipelines ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass pipeline registration for AMDGCN transforms.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// RegAlloc Pipeline
//===----------------------------------------------------------------------===//

/// Options for the RegAlloc pass pipeline.
struct RegAllocPipelineOptions
    : public PassPipelineOptions<RegAllocPipelineOptions> {
  mlir::detail::PassOptions::Option<std::string> buildMode{
      *this, "mode",
      llvm::cl::desc("Graph build mode: \"minimal\" (default) or \"full\""),
      llvm::cl::init("minimal")};
};

/// Build the RegAlloc pass pipeline.
///
/// This pipeline performs register allocation for AMDGCN kernels by running
/// the following passes in sequence:
/// 1. Bufferization - inserts copies to remove potentially clobbered values,
/// and removes phi-nodes arguments with register value semantics.
/// 2. ToRegisterSemantics - converts value allocas to unallocated register
///    semantics
/// 3. RegisterAlloc - performs the actual register allocation
static void buildRegAllocPassPipeline(OpPassManager &pm,
                                      const RegAllocPipelineOptions &options) {
  pm.addPass(createAMDGCNBufferization());
  pm.addPass(createToRegisterSemantics());
  pm.addPass(createRegisterDCE());
  RegisterColoringOptions coloringOpts;
  coloringOpts.buildMode = options.buildMode;
  pm.addPass(createRegisterColoring(coloringOpts));
  pm.addPass(createHoistOps());
}

void mlir::aster::amdgcn::registerRegAllocPassPipeline() {
  PassPipelineRegistration<RegAllocPipelineOptions>(
      "amdgcn-reg-alloc", "Run the AMDGCN register allocation pipeline",
      buildRegAllocPassPipeline);
}
