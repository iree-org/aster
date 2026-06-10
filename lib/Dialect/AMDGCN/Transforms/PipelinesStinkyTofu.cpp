//===- PipelinesStinkyTofu.cpp - StinkyTofu handoff pipelines ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PipelinesInternal.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

static void buildAMDGCNHandoffBackendPassPipeline(
    OpPassManager &pm, const AMDGCNBackendPipelineOptions &options) {
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
    // Scheduling phase, mirroring buildAMDGCNBackendPassPipeline. StinkyTofu
    // schedules + inserts waits downstream, so ASTER's low-level scheduler is
    // off by default (ll-sched=0); the gate stays for A/B experiments. LDS
    // allocation now runs late, before regalloc, so the materialized
    // get_lds_offset alloca is colored.
    if (options.hoistIterArgWaits) {
      kernelPm.addPass(createHoistIterArgWaits());
      kernelPm.addPass(createCanonicalizerPass());
    }
    if (options.llSched > 0) {
      LowLevelSchedulerOptions llSchedOpts;
      llSchedOpts.preset = options.llSched;
      kernelPm.addPass(createLowLevelScheduler(llSchedOpts));
    }
    buildLdsAllocPassPipeline(kernelPm);
    RegAllocPipelineOptions regAllocOpts;
    regAllocOpts.numVGPRs = options.numVGPRs;
    regAllocOpts.numAGPRs = options.numAGPRs;
    regAllocOpts.hoistIterArgWaits = options.hoistIterArgWaits;
    buildRegAllocPassPipeline(kernelPm, regAllocOpts);
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
  }
  pm.addPass(createAMDGCNStripWaits());
  {
    OpPassManager &kernelPm = pm.nest<amdgcn::ModuleOp>().nest<KernelOp>();
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
    kernelPm.addPass(createLegalizeCF());
    kernelPm.addPass(createCanonicalizerPass());
    kernelPm.addPass(createCSEPass());
  }
  {
    SetNormalFormsOptions nfOpts;
    nfOpts.moduleForms = {"no_lsir_ops", "no_lsir_control_ops"};
    pm.addPass(createSetNormalForms(nfOpts));
  }
}

static void registerAMDGCNHandoffBackendPassPipeline() {
  PassPipelineRegistration<AMDGCNBackendPipelineOptions>(
      "amdgcn-handoff-backend",
      "Run the AMDGCN handoff backend pipeline (reg-alloc kept, waits "
      "stripped) for a downstream scheduler+waiter",
      buildAMDGCNHandoffBackendPassPipeline);
}

void mlir::aster::amdgcn::registerStinkyTofuPipelines() {
  registerAMDGCNHandoffBackendPassPipeline();
}
