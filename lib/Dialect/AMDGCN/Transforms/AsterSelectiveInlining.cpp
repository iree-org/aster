//===- AsterSelectiveInlining.cpp - Selective Inlining Pass --------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_ASTERSELECTIVEINLINING
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AsterSelectiveInlining pass
//===----------------------------------------------------------------------===//
struct AsterSelectiveInlining
    : public amdgcn::impl::AsterSelectiveInliningBase<AsterSelectiveInlining> {
public:
  using Base::Base;

  void runOnOperation() override;

  /// Helper function to run a nested pass pipeline
  static LogicalResult runPipelineHelper(Pass &pass, OpPassManager &pipeline,
                                         Operation *op) {
    return mlir::cast<AsterSelectiveInlining>(pass).runPipeline(pipeline, op);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AsterSelectiveInlining pass implementation
//===----------------------------------------------------------------------===//

void AsterSelectiveInlining::runOnOperation() {
  Operation *op = getOperation();

  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    op->emitOpError() << " was scheduled to run under the inliner, but does "
                         "not define a symbol table";
    return signalPassFailure();
  }

  CallGraph &cg = getAnalysis<CallGraph>();

  // Default inliner optimization pipeline (canonicalize)
  auto defaultPipeline = [](OpPassManager &pm) {
    pm.addPass(createCanonicalizerPass());
  };
  InlinerConfig config(defaultPipeline, /*maxInliningIterations=*/4);

  // Create profitability callback that checks for sched.* attributes
  bool allowScheduled = allowScheduledCalls;
  auto profitabilityCb =
      [allowScheduled](const Inliner::ResolvedCall &resolvedCall) {
        Operation *callOp = resolvedCall.call;
        if (!callOp)
          return true;
        if (hasSchedAttribute(callOp))
          return allowScheduled;
        return true;
      };

  Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper,
                  config, profitabilityCb);
  if (failed(inliner.doInlining()))
    signalPassFailure();
}
