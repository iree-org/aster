//===- AttachScheduler.cpp - Attach scheduler attribute to functions ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_ATTACHSCHEDULER
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AttachScheduler pass
//===----------------------------------------------------------------------===//

struct AttachScheduler
    : public amdgcn::impl::AttachSchedulerBase<AttachScheduler> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void AttachScheduler::runOnOperation() {
  if (path.empty()) {
    getOperation()->emitError(
        "amdgcn-attach-scheduler: --path option is required");
    return signalPassFailure();
  }

  MLIRContext *ctx = &getContext();
  GenericSchedLabelerAttr labelerAttr = GenericSchedLabelerAttr::get(ctx, path);
  if (labelerAttr.getLabeler().isTrivial())
    return signalPassFailure();

  aster_utils::GenericSchedulerAttr schedAttr =
      aster_utils::GenericSchedulerAttr::get(
          ctx, amdgcn::ValueSchedulerAttr::get(ctx), labelerAttr,
          aster_utils::StageTopoSortSchedAttr::get(ctx));

  getOperation()->walk([&](FunctionOpInterface funcOp) {
    if (funcOp->getDiscardableAttr("aster.sched"))
      return;
    funcOp->setDiscardableAttr("aster.sched", schedAttr);
  });
}
