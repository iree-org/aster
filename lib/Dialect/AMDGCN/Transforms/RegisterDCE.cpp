//===- RegisterDCE.cpp - Dead code elimination for register copies
//---------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "amdgcn-register-dce"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REGISTERDCE
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// RegisterDCE pass
//===----------------------------------------------------------------------===//
struct RegisterDCE : public amdgcn::impl::RegisterDCEBase<RegisterDCE> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static Value getTargetRegister(Operation *op) {
  if (auto copyOp = dyn_cast<lsir::CopyOp>(op))
    return copyOp.getTarget();

  if (auto instOp = dyn_cast<inst::VOP1Op>(op);
      instOp && instOp.getOpcode() == OpCode::V_MOV_B32_E32) {
    return instOp.getVdst();
  }
  if (auto instOp = dyn_cast<inst::SOP1Op>(op);
      instOp && instOp.getOpcode() == OpCode::S_MOV_B32) {
    return instOp.getSdst();
  }
  return nullptr;
}

void amdgcn::registerDCE(Operation *op, DataFlowSolver &solver) {
  op->walk([&](Operation *operation) {
    Value target = getTargetRegister(operation);

    if (!target)
      return;

    // Get the liveness state after the operation.
    const auto *state = solver.lookupState<LivenessState>(
        solver.getProgramPointAfter(operation));
    const LivenessState::ValueSet *liveValues =
        state ? state->getLiveValues() : nullptr;
    if (!liveValues)
      return;

    // Resolve the target to underlying allocas. For make_register_range
    // targets, the liveness analysis tracks individual allocas, not the
    // range value itself. We must check those allocas for liveness.
    FailureOr<ValueRange> allocas = getAllocasOrFailure(target);
    if (failed(allocas))
      return; // Conservative: preserve operations we can't analyze.

    // The operation is live if any of its target allocas are live.
    bool isLive = llvm::any_of(
        *allocas, [&](Value alloca) { return liveValues->contains(alloca); });
    if (isLive)
      return;

    operation->erase();
  });
}

//===----------------------------------------------------------------------===//
// RegisterDCE pass
//===----------------------------------------------------------------------===//
void RegisterDCE::runOnOperation() {
  Operation *op = getOperation();

  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<LivenessAnalysis>(symbolTable);

  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "failed to run LivenessAnalysis";
    return signalPassFailure();
  }

  registerDCE(op, solver);
}
