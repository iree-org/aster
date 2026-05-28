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
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
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
  mlir::detail::PassOptions::Option<int32_t> numVGPRs{
      *this, "num-vgprs",
      llvm::cl::desc("Maximum VGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numAGPRs{
      *this, "num-agprs",
      llvm::cl::desc("Maximum AGPRs for allocation (default 256)"),
      llvm::cl::init(256)};
  mlir::detail::PassOptions::Option<int32_t> numSGPRs{
      *this, "num-sgprs",
      llvm::cl::desc("Maximum SGPRs for allocation (default 102)"),
      llvm::cl::init(102)};
  mlir::detail::PassOptions::Option<bool> hoistIterArgWaits{
      *this, "hoist-iter-arg-waits",
      llvm::cl::desc("Hoist iter_arg-dependent waits to loop head"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<std::string> regAllocSolver{
      *this, "reg-alloc-solver",
      llvm::cl::desc(
          "Register allocation back-end: \"greedy\" (default) or \"ilp\""),
      llvm::cl::init("greedy")};
  mlir::detail::PassOptions::Option<int32_t> ilpTimeLimitMs{
      *this, "ilp-time-limit-ms",
      llvm::cl::desc("Wall-clock time limit for the ILP solver in milliseconds "
                     "(default 100000)"),
      llvm::cl::init(100000)};
  mlir::detail::PassOptions::Option<std::string> ilpObjective{
      *this, "ilp-objective",
      llvm::cl::desc("ILP objective: \"min-pressure\" (default) or "
                     "\"feasibility\""),
      llvm::cl::init("min-pressure")};
};

/// Build the RegAlloc pass pipeline.
///
/// This pipeline performs register allocation for AMDGCN kernels by running
/// the following passes in sequence:
/// 1. Bufferization - inserts copies to remove potentially clobbered values,
///    and removes phi-node arguments with register value semantics.
/// 2. ToRegisterSemantics - converts value allocas to unallocated register
///    semantics.
/// 3. RegisterColoring - performs the actual register allocation.
/// 4. PostRegAllocLegalization - expands lsir.copy to hardware mov instructions
///    and erases redundant movs whose source and destination are the same
///    physical register.
static void buildRegAllocPassPipeline(OpPassManager &pm,
                                      const RegAllocPipelineOptions &options) {
  // The low-level scheduler and LDS allocation run upstream of this pipeline
  // (buildAMDGCNBackendPassPipeline runs the scheduler, then
  // buildLdsAllocPassPipeline), so register allocation sees folded LDS offsets.
  pm.addPass(createAMDGCNBufferization());
  if (options.hoistIterArgWaits) {
    pm.addPass(createHoistIterArgWaits());
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(createToRegisterSemantics());
  // Post-condition of to-register-semantics is now enforced by
  // KernelOp::verifyRegions() via the normal_forms attribute set by the pass.
  pm.addPass(createRegisterDCE());
  pm.addPass(createPreColoringLegalization());
  RegisterColoringOptions coloringOpts;
  coloringOpts.buildMode = options.buildMode;
  coloringOpts.numVGPRs = options.numVGPRs;
  coloringOpts.numAGPRs = options.numAGPRs;
  coloringOpts.numSGPRs = options.numSGPRs;
  coloringOpts.regAllocSolver = options.regAllocSolver;
  coloringOpts.ilpTimeLimitMs = options.ilpTimeLimitMs;
  coloringOpts.ilpObjective = options.ilpObjective;
  pm.addPass(createRegisterColoring(coloringOpts));
  pm.addPass(createPostRegAllocLegalization());
  pm.addPass(createHoistOps());
  pm.addPass(createCFGSimplification());
}

static void registerRegAllocPassPipeline() {
  PassPipelineRegistration<RegAllocPipelineOptions>(
      "amdgcn-reg-alloc", "Run the AMDGCN register allocation pipeline",
      buildRegAllocPassPipeline);
}

//===----------------------------------------------------------------------===//
// LDS Allocation Pipeline
//===----------------------------------------------------------------------===//

/// Build the LDS allocation pipeline.
///
/// Runs after the low-level scheduler (see buildAMDGCNBackendPassPipeline) so
/// the scheduler sees live amdgcn.alloc_lds / amdgcn.get_lds_offset handles:
/// the memory-dependence analysis traces LDS accesses to their originating
/// alloc_lds to order same-buffer LDS accesses, which folded constants make
/// impossible.
/// This mirrors register allocation, the scheduler reasons about the resource,
/// then allocation assigns it.
///
/// amdgcn-lds-alloc assigns byte offsets (and the kernel shared_memory_size).
/// amdgcn-convert-lds-buffers folds get_lds_offset to constants and erases the
/// handles.
static void buildLdsAllocPassPipeline(OpPassManager &pm) {
  pm.addPass(createLDSAlloc());
  pm.addPass(createConvertLDSBuffers());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

static void registerLdsAllocationPassPipeline() {
  PassPipelineRegistration<>(
      "amdgcn-lds-alloc-pipeline",
      "Run the AMDGCN LDS allocation pipeline (assign byte offsets, fold "
      "get_lds_offset to constants, SROA/mem2reg cleanup)",
      buildLdsAllocPassPipeline);
}

//===----------------------------------------------------------------------===//
// LateWaits Pipeline
//===----------------------------------------------------------------------===//

static void buildLateWaitsPassPipeline(OpPassManager &pm) {
  pm.nest<amdgcn::ModuleOp>().addPass(createWaitInsertion());
  pm.addPass(mlir::createMem2Reg());
  pm.nest<amdgcn::ModuleOp>().addPass(createAMDGCNConvertWaits({true}));
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
  mlir::detail::PassOptions::Option<int32_t> llSched{
      *this, "ll-sched",
      llvm::cl::desc("Low-level scheduler preset (0 = off, 1+ = run with "
                     "preset N; see SchedAttrs.cpp)"),
      llvm::cl::init(0)};
  mlir::detail::PassOptions::Option<bool> setMfmaPriority{
      *this, "set-mfma-priority",
      llvm::cl::desc("Insert s_setprio around MFMA groups"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<std::string> regAllocSolver{
      *this, "reg-alloc-solver",
      llvm::cl::desc(
          "Register allocation back-end: \"greedy\" (default) or \"ilp\""),
      llvm::cl::init("greedy")};
  mlir::detail::PassOptions::Option<int32_t> ilpTimeLimitMs{
      *this, "ilp-time-limit-ms",
      llvm::cl::desc("Wall-clock time limit for the ILP solver in milliseconds "
                     "(default 100000)"),
      llvm::cl::init(100000)};
  mlir::detail::PassOptions::Option<std::string> ilpObjective{
      *this, "ilp-objective",
      llvm::cl::desc("ILP objective: \"min-pressure\" (default) or "
                     "\"feasibility\""),
      llvm::cl::init("min-pressure")};
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
    // Scheduling phase: hoist iter-arg waits then run the low-level scheduler
    // BEFORE LDS allocation, so the scheduler sees live
    // alloc_lds/get_lds_offset.
    if (options.hoistIterArgWaits) {
      kernelPm.addPass(createHoistIterArgWaits());
      kernelPm.addPass(createCanonicalizerPass());
    }
    if (options.llSched > 0) {
      LowLevelSchedulerOptions llSchedOpts;
      llSchedOpts.preset = options.llSched;
      kernelPm.addPass(createLowLevelScheduler(llSchedOpts));
    }
    // LDS allocation: assign byte offsets + fold the handles to constants,
    // after the scheduler, before register allocation.
    buildLdsAllocPassPipeline(kernelPm);
    RegAllocPipelineOptions regAllocOpts;
    regAllocOpts.numVGPRs = options.numVGPRs;
    regAllocOpts.numAGPRs = options.numAGPRs;
    regAllocOpts.hoistIterArgWaits = options.hoistIterArgWaits;
    regAllocOpts.regAllocSolver = options.regAllocSolver;
    regAllocOpts.ilpTimeLimitMs = options.ilpTimeLimitMs;
    regAllocOpts.ilpObjective = options.ilpObjective;
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
    if (options.setMfmaPriority)
      kernelPm.addPass(createSetMFMAPriority());
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
  registerLdsAllocationPassPipeline();
  registerLateWaitsPassPipeline();
  registerAMDGCNBackendPassPipeline();
}
