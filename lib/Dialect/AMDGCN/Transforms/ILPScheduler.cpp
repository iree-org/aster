//===- ILPScheduler.cpp - ILP-based pre-RA instruction scheduler ----------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-register-allocation instruction scheduler that assigns each instruction a
// firing time via an ILP (CP-SAT) model minimizing issue latency. Reuses the
// dependency graph (ValueSchedulerAttr) and replaces only the ordering policy
// (ILPSchedulerAttr) relative to the greedy low-level scheduler.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNILPSCHEDULER
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::aster_utils;

namespace {

struct AMDGCNILPSchedulerPass
    : public amdgcn::impl::AMDGCNILPSchedulerBase<AMDGCNILPSchedulerPass> {
  using Base::Base;

  void runOnOperation() override {
    KernelOp kernel = getOperation();
    MLIRContext *ctx = kernel.getContext();

    auto allInlined = AllInlinedAttr::get(ctx);
    if (failed(allInlined.checkOperation(kernel).checkAndReport()))
      return signalPassFailure();

    // The sched interface implementations are external models, so an explicit
    // cast through the interface is required here. The graph
    // (ValueSchedulerAttr) is reused unchanged; only the builder
    // (ILPSchedulerAttr) differs from the greedy low-level scheduler.
    GenericSchedulerAttr compositeAttr = GenericSchedulerAttr::get(
        ctx, mlir::cast<SchedGraphAttrInterface>(ValueSchedulerAttr::get(ctx)),
        SchedListLabelerAttr::get(ctx, ArrayRef<SchedLabelerAttrInterface>{}),
        mlir::cast<SchedBuilderAttrInterface>(ILPSchedulerAttr::get(
            ctx, level, ilpTimeLimitMs, mfmaGap, vmemGap, lgkmGap,
            barrierBypass, maxLoadDistance, windowMfmas, minLgkmDistance)));

    StringAttr schedName = StringAttr::get(ctx, "amdgcn.ilp_sched");
    SmallVector<SchedInfo> schedInfos(
        {SchedInfo{kernel, schedName, compositeAttr, 0}});

    ISAVersion isaVersion =
        getIsaForOp(cast<amdgcn::ModuleOp>(kernel->getParentOp()));
    AnalysisManager analysisManager = getAnalysisManager();
    if (failed(applyScheds(kernel, schedInfos, analysisManager, isaVersion)))
      return signalPassFailure();
  }
};

} // namespace
