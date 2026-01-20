//===- LivenessAnalysis.cpp - Liveness analysis ---------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "liveness-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// LivenessState
//===----------------------------------------------------------------------===//

void LivenessState::print(raw_ostream &os) const {
  if (isEmpty()) {
    os << "[]";
    return;
  }
  if (isTop()) {
    os << "<top>";
    return;
  }
  const LiveSet *eqClassIds = getLiveEqClassIds();
  assert(eqClassIds && "Live equivalence classes should be valid here");
  os << "[";
  llvm::interleaveComma(*eqClassIds, os,
                        [&](EqClassID eqClassId) { os << eqClassId; });
  os << "]";
}

ChangeResult LivenessState::meet(const LivenessState &lattice) {
  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  if (isTop())
    return ChangeResult::NoChange;

  if (lattice.isTop())
    return setToTop();

  if (isEmpty()) {
    liveEqClasses = *lattice.liveEqClasses;
    return ChangeResult::Change;
  }
  const LiveSet *latticeEqClasses = lattice.getLiveEqClassIds();
  assert(latticeEqClasses && "Lattice equivalence classes should be valid");
  return appendEqClassIds(*latticeEqClasses);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

ArrayRef<EqClassID> LivenessAnalysis::getEqClassIds(Value v) const {
  if (!isa<RegisterTypeInterface>(v.getType()))
    return {};
  return aliasAnalysis->getEqClassIds(v);
}

/// Transfer function for liveness analysis.
void LivenessAnalysis::transferFunction(const LivenessState &after,
                                        LivenessState *before,
                                        SmallVector<EqClassID> &&deadEqClassIds,
                                        ArrayRef<EqClassID> liveEqClassIds) {
  SmallVector<EqClassID> resultLiveIds;
  SmallVector<EqClassID> afterIds = llvm::to_vector(*after.getLiveEqClassIds());
  if (!afterIds.empty())
    llvm::sort(afterIds);
  if (!deadEqClassIds.empty())
    llvm::sort(deadEqClassIds);
  std::set_difference(afterIds.begin(), afterIds.end(), deadEqClassIds.begin(),
                      deadEqClassIds.end(), std::back_inserter(resultLiveIds));
  llvm::append_range(resultLiveIds, liveEqClassIds);
  propagateIfChanged(before, before->appendEqClassIds(resultLiveIds));
}

bool LivenessAnalysis::handleTopPropagation(const LivenessState &after,
                                            LivenessState *before) {
  if (after.isTop() || before->isTop()) {
    propagateIfChanged(before, before->setToTop());
    return true;
  }
  return false;
}

#define DUMP_STATE_HELPER(name, obj)                                           \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "Visiting " name ": " << obj << "\n";                              \
      os << "  Incoming lattice: ";                                            \
      after.print(os);                                                         \
      os << "\n  Outgoing lattice: ";                                          \
      before->print(os);                                                       \
    });                                                                        \
  });

/// Helper to collect equivalence class IDs from a range of values.
static void collectEqClassIds(DPSAliasAnalysis *aliasAnalysis,
                              ValueRange values,
                              SmallVectorImpl<EqClassID> &eqClassIds) {
  for (Value v : values) {
    if (!isa<RegisterTypeInterface>(v.getType()))
      continue;
    llvm::append_range(eqClassIds, aliasAnalysis->getEqClassIds(v));
  }
}

LogicalResult LivenessAnalysis::visitOperation(Operation *op,
                                               const LivenessState &after,
                                               LivenessState *before) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return success();

  // Handle instruction operations.
  if (auto inst = dyn_cast<InstOpInterface>(op)) {
    SmallVector<EqClassID> deadEqClassIds;
    collectEqClassIds(aliasAnalysis, op->getResults(), deadEqClassIds);
    // Append instruction outputs to dead values, as they are actually result
    // values.
    collectEqClassIds(aliasAnalysis, inst.getInstOuts(), deadEqClassIds);
    SmallVector<EqClassID> liveEqClassIds;
    collectEqClassIds(aliasAnalysis, inst.getInstIns(), liveEqClassIds);
    transferFunction(after, before, std::move(deadEqClassIds), liveEqClassIds);
    return success();
  }

  // Handle MakeRegisterRangeOp.
  if (auto mOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
    SmallVector<EqClassID> deadEqClassIds;
    collectEqClassIds(aliasAnalysis, op->getResults(), deadEqClassIds);
    // The operands are aliased with the result, they must be live before this
    // op.
    SmallVector<EqClassID> liveEqClassIds;
    collectEqClassIds(aliasAnalysis, op->getOperands(), liveEqClassIds);
    transferFunction(after, before, std::move(deadEqClassIds), liveEqClassIds);
    return success();
  }

  // Handle SplitRegisterRangeOp.
  if (auto sOp = dyn_cast<amdgcn::SplitRegisterRangeOp>(op)) {
    SmallVector<EqClassID> deadEqClassIds;
    collectEqClassIds(aliasAnalysis, op->getResults(), deadEqClassIds);
    // The operands are aliased with the results, they must be live before this
    // op.
    SmallVector<EqClassID> liveEqClassIds;
    collectEqClassIds(aliasAnalysis, op->getOperands(), liveEqClassIds);
    transferFunction(after, before, std::move(deadEqClassIds), liveEqClassIds);
    return success();
  }

  // Handle RegInterferenceOp.
  if (auto iOp = dyn_cast<amdgcn::RegInterferenceOp>(op)) {
    // Reg interference operations do not affect liveness.
    transferFunction(after, before, {}, {});
    return success();
  }

  // Handle generic operations.
  SmallVector<EqClassID> deadEqClassIds;
  collectEqClassIds(aliasAnalysis, op->getResults(), deadEqClassIds);
  SmallVector<EqClassID> liveEqClassIds;
  collectEqClassIds(aliasAnalysis, op->getOperands(), liveEqClassIds);
  transferFunction(after, before, std::move(deadEqClassIds), liveEqClassIds);
  return success();
}

void LivenessAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const LivenessState &after,
                                          LivenessState *before) {
  DUMP_STATE_HELPER("block", block);
  if (handleTopPropagation(after, before))
    return;
  SmallVector<EqClassID> deadEqClassIds;
  collectEqClassIds(aliasAnalysis, successor->getArguments(), deadEqClassIds);
  transferFunction(after, before, std::move(deadEqClassIds), {});
}

void LivenessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const LivenessState &after, LivenessState *before) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return;
  assert(action == dataflow::CallControlFlowAction::ExternalCallee &&
         "we don't support inter-procedural analysis");
  SmallVector<EqClassID> deadEqClassIds;
  collectEqClassIds(aliasAnalysis, call->getResults(), deadEqClassIds);
  SmallVector<EqClassID> liveEqClassIds;
  collectEqClassIds(aliasAnalysis, call.getArgOperands(), liveEqClassIds);
  transferFunction(after, before, std::move(deadEqClassIds), liveEqClassIds);
}

void LivenessAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionSuccessor regionTo, const LivenessState &after,
    LivenessState *before) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return;
  SmallVector<EqClassID> deadEqClassIds;
  collectEqClassIds(aliasAnalysis, regionTo.getSuccessorInputs(),
                    deadEqClassIds);
  transferFunction(after, before, std::move(deadEqClassIds), {});
}

void LivenessAnalysis::setToExitState(LivenessState *lattice) {
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

#undef DUMP_STATE_HELPER
