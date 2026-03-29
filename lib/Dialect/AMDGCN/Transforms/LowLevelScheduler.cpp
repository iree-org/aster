//===- LowLevelScheduler.cpp - Pre-RA instruction scheduler ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-register-allocation instruction scheduler that models AMD GPU hardware
// execution queues (VALU, XDL, SALU, VMEM, LGKM). Reorders instructions
// within basic blocks to hide issue latency using a greedy algorithm.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LOWLEVELSCHEDULER
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::aster_utils;

namespace {

struct LowLevelSchedulerPass
    : public amdgcn::impl::LowLevelSchedulerBase<LowLevelSchedulerPass> {
  using Base::Base;

  void runOnOperation() override {
    KernelOp kernel = getOperation();
    MLIRContext *ctx = kernel.getContext();

    auto allInlined = AllInlinedAttr::get(ctx);
    if (failed(normalform::verifyNormalForm(kernel, allInlined,
                                            /*emitDiagnostics=*/true)))
      return signalPassFailure();

    GenericSchedulerAttr compositeAttr = GenericSchedulerAttr::get(
        ctx, ValueSchedulerAttr::get(ctx),
        SchedListLabelerAttr::get(ctx, ArrayRef<SchedLabelerAttrInterface>{}),
        LowLevelSchedulerAttr::get(ctx, debugStalls));

    StringAttr schedName = StringAttr::get(ctx, "amdgcn.low_level_sched");
    SmallVector<SchedInfo> schedInfos(
        {SchedInfo{kernel, schedName, compositeAttr, 0}});

    AnalysisManager analysisManager = getAnalysisManager();
    if (failed(applyScheds(kernel, schedInfos, analysisManager)))
      return signalPassFailure();
  }
};

} // namespace
