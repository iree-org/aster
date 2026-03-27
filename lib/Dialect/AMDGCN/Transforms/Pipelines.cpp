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

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::aster;
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
  mlir::detail::PassOptions::Option<bool> optimize{
      *this, "optimize",
      llvm::cl::desc("Run optimizeGraph to coalesce non-interfering nodes"),
      llvm::cl::init(true)};
  mlir::detail::PassOptions::Option<int32_t> numVGPRs{
      *this, "num-vgprs",
      llvm::cl::desc("Maximum VGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numAGPRs{
      *this, "num-agprs",
      llvm::cl::desc("Maximum AGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<bool> hoistIterArgWaits{
      *this, "hoist-iter-arg-waits",
      llvm::cl::desc("Hoist iter_arg-dependent waits to loop head"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<bool> llSched{
      *this, "ll-sched",
      llvm::cl::desc("Run low-level scheduler before register semantics"),
      llvm::cl::init(false)};
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
  if (options.hoistIterArgWaits) {
    pm.addPass(createHoistIterArgWaits());
    pm.addPass(createCanonicalizerPass());
  }
  if (options.llSched)
    pm.addPass(createLowLevelScheduler());
  pm.addPass(createAMDGCNBufferization());
  if (options.hoistIterArgWaits) {
    pm.addPass(createHoistIterArgWaits());
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(createToRegisterSemantics());
  // Post-condition of to-register-semantics is now enforced by
  // KernelOp::verifyRegions() via the normal_forms attribute set by the pass.
  pm.addPass(createRegisterDCE());
  RegisterColoringOptions coloringOpts;
  coloringOpts.buildMode = options.buildMode;
  coloringOpts.optimize = options.optimize;
  coloringOpts.numVGPRs = options.numVGPRs;
  coloringOpts.numAGPRs = options.numAGPRs;
  pm.addPass(createRegisterColoring(coloringOpts));
  pm.addPass(createHoistOps());
  pm.addPass(createCFGSimplification());
}

static void registerRegAllocPassPipeline() {
  PassPipelineRegistration<RegAllocPipelineOptions>(
      "amdgcn-reg-alloc", "Run the AMDGCN register allocation pipeline",
      buildRegAllocPassPipeline);
}

//===----------------------------------------------------------------------===//
// LateWaits Pipeline
//===----------------------------------------------------------------------===//

static void buildLateWaitsPassPipeline(OpPassManager &pm) {
  pm.addPass(createWaitInsertion());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(createAMDGCNConvertWaits({true}));
}

static void registerLateWaitsPassPipeline() {
  PassPipelineRegistration<>("amdgcn-late-waits",
                             "Run the late wait insertion pipeline (insert "
                             "waits, mem2reg, convert waits)",
                             buildLateWaitsPassPipeline);
}

//===----------------------------------------------------------------------===//
// AMDGCN Backend Pipeline
//===----------------------------------------------------------------------===//

struct AMDGCNBackendPipelineOptions
    : public PassPipelineOptions<AMDGCNBackendPipelineOptions> {
  mlir::detail::PassOptions::Option<int32_t> numVGPRs{
      *this, "num-vgprs",
      llvm::cl::desc("Maximum VGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numAGPRs{
      *this, "num-agprs",
      llvm::cl::desc("Maximum AGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<bool> hoistIterArgWaits{
      *this, "hoist-iter-arg-waits",
      llvm::cl::desc("Hoist iter_arg-dependent waits to loop head"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<bool> llSched{
      *this, "ll-sched",
      llvm::cl::desc("Run low-level scheduler before register allocation"),
      llvm::cl::init(false)};
};

static void
buildAMDGCNBackendPassPipeline(OpPassManager &pm,
                               const AMDGCNBackendPipelineOptions &options) {
  // Assert no LSIR compute/memory ops remain at backend entry.
  // Only lsir.cmpi/cmpf/select survive (lowered by LegalizeCF later).
  {
    SetNormalFormsOptions nfOpts;
    nfOpts.moduleForms = {"no_lsir_compute_ops"};
    pm.addPass(createSetNormalForms(nfOpts));
  }
  {
    OpPassManager &kernelPm = pm.nest<amdgcn::ModuleOp>().nest<KernelOp>();
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
    kernelPm.addPass(createExpandMetadataOps());
    kernelPm.addPass(createLegalizeOperands());
    RegAllocPipelineOptions regAllocOpts;
    regAllocOpts.numVGPRs = options.numVGPRs;
    regAllocOpts.numAGPRs = options.numAGPRs;
    regAllocOpts.hoistIterArgWaits = options.hoistIterArgWaits;
    regAllocOpts.llSched = options.llSched;
    buildRegAllocPassPipeline(kernelPm, regAllocOpts);
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
  }
  buildLateWaitsPassPipeline(pm);
  {
    OpPassManager &kernelPm = pm.nest<amdgcn::ModuleOp>().nest<KernelOp>();
    // Canonicalize + CSE before LegalizeCF to deduplicate allocas.
    // After register coloring, non-interfering values colored to the same
    // physical register may have separate allocas of the same type.
    // CSE merges them so LegalizeCF sees exactly one alloca per register.
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
    kernelPm.addPass(createLegalizeCF());
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
  }
  // Assert all LSIR ops are gone. LegalizeCF lowered the last ones
  // (lsir.cmpi, lsir.cmpf, lsir.select).
  {
    SetNormalFormsOptions nfOpts;
    nfOpts.moduleForms = {"no_lsir_ops", "no_lsir_control_ops"};
    pm.addPass(createSetNormalForms(nfOpts));
  }
}

static void registerAMDGCNBackendPassPipeline() {
  PassPipelineRegistration<AMDGCNBackendPipelineOptions>(
      "amdgcn-backend",
      "Run the AMDGCN backend pipeline (canonicalize, cse, reg-alloc, "
      "late-waits, legalize cf, canonicalize, cse)",
      buildAMDGCNBackendPassPipeline);
}

void mlir::aster::amdgcn::registerPipelines() {
  registerRegAllocPassPipeline();
  registerLateWaitsPassPipeline();
  registerAMDGCNBackendPassPipeline();
}
