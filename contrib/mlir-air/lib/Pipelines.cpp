// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Pipelines.cpp - mlir-air pass pipelines ----------------------------===//

#include "aster/CodeGen/Passes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

static void addSROA(OpPassManager &pm) {
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createSROA());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(amdgcn::createMem2Reg());
  pm.addPass(aster_utils::createAsterSelectiveInlining(
      {/*allowScheduledCalls=*/true}));
}

static void addPostSROACleanups(OpPassManager &pm) {
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(aster::createConstexprExpansion());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(aster::createSimplifyAllocaIterArgs());
  pm.addPass(aster::createDecomposeMemrefIterArgs());
  pm.addPass(aster_utils::createDestructureStructIterArgs());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createSROA());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(amdgcn::createMem2Reg());
  pm.addPass(aster::createForwardStoreToLoad());
  pm.addPass(aster::createPromoteLoopCarriedMemrefs());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

static void buildMlirAirToAsmPipeline(OpPassManager &pm) {
  // PRE_SCHEDULING_CLEANUP
  pm.addPass(aster_utils::createAsterSelectiveInlining());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // CONSTEXPR_EXPANSION
  pm.addPass(aster::createConstexprExpansion());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createSROA());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(amdgcn::createMem2Reg());
  pm.addPass(aster::createForwardStoreToLoad());
  pm.addPass(aster::createPromoteLoopCarriedMemrefs());
  pm.addPass(createCanonicalizerPass());

  // SROA + POST_SROA_CLEANUPS
  addSROA(pm);
  addPostSROACleanups(pm);

  // CONVERT_LDS_BUFFERS
  pm.addPass(createLDSAlloc());
  pm.addPass(createConvertLDSBuffers());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // PHASE_LOWER_TO_AMDGCN
  pm.addPass(aster::createLegalizer());
  pm.addPass(mlir::affine::createAffineExpandIndexOpsAsAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(aster_utils::createExpandAffineApply());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(createCSEPass());
  pm.addPass(aster_utils::createDecomposeByLoopInvariant());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(aster_utils::createDecomposeByCSE());
  pm.addPass(createCSEPass());
  pm.addPass(aster_utils::createRaiseToAffine());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(createCSEPass());
  pm.addPass(aster::createAffineOptimizePtrAdd({true}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(aster::createFactorizeAffineExpr());
  pm.addPass(aster::createToIntArith());
  {
    aster_utils::RemoveAssumeOpsOptions opts;
    opts.removePassthrough = true;
    pm.addPass(aster_utils::createRemoveAssumeOps(opts));
  }
  pm.addPass(aster::createOptimizeArith());
  pm.addPass(aster_utils::createOptimizePtrAdd());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(aster_utils::createResolveAnyIterArgs());
  pm.addPass(createSetABI());
  pm.addPass(createConvertSCFControlFlow());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(aster::createCodeGen());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createOptimizeAMDGCN());
  pm.addPass(createToAMDGCN());
  {
    OpPassManager &kernelPm = pm.nest<amdgcn::ModuleOp>().nest<KernelOp>();
    kernelPm.addPass(createHoistOps());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // amdgcn-backend (registered pipeline)
  (void)parsePassPipeline("amdgcn-backend", pm);

  // NOP insertion
  pm.addPass(createRemoveTestInst());
  pm.addPass(createAMDGCNHazards());
}

} // namespace

namespace mlir::aster::mlir_air {
void registerPipelines() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  PassPipelineRegistration<>(
      "mlir-air-to-asm",
      "Full mlir-air pipeline: SROA, legalizer, codegen, backend",
      buildMlirAirToAsmPipeline);
}
} // namespace mlir::aster::mlir_air
