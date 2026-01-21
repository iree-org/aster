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

/// Helper to print a block as an operand (e.g. "^bb3").
struct PrintableBlock {
  Block *block;
  friend raw_ostream &operator<<(raw_ostream &os, const PrintableBlock &b) {
    b.block->printAsOperand(os);
    return os;
  }
};

/// Helper to print a ProgramPoint* by dereferencing it.
struct PrintableProgramPoint {
  ProgramPoint *point;
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PrintableProgramPoint &p) {
    p.point->print(os);
    return os;
  }
};

#define DUMP_STATE_HELPER(uniqueGuardName, name, obj)                          \
  auto uniqueGuardName = llvm::make_scope_exit([&]() {                         \
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
  DUMP_STATE_HELPER(g1, "op", OpWithFlags(op, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return success();

  // AllocaOp is the only operation that defines a new equivalence class.
  // As we traverse dataflow backwards, the equivalence class is dead before.
  if (auto allocaOp = dyn_cast<amdgcn::AllocaOp>(op)) {
    SmallVector<EqClassID> deadEqClassIds;
    collectEqClassIds(aliasAnalysis, allocaOp.getResult(), deadEqClassIds);
    transferFunction(after, before, std::move(deadEqClassIds), {});
    return success();
  }

  // Handle MakeRegisterRangeOp, SplitRegisterRangeOp and RegInterferenceOp.
  // These are pure aliasing ops and do not affect liveness.
  if (isa<amdgcn::MakeRegisterRangeOp, amdgcn::SplitRegisterRangeOp,
          amdgcn::RegInterferenceOp>(op)) {
    transferFunction(after, before, {}, {});
    return success();
  }

  // Handle InstOpInterface operations.
  // These operations can only make existing equivalence classes live before.
  if (auto inst = dyn_cast<InstOpInterface>(op)) {
    SmallVector<EqClassID> liveEqClassIds;
    // collectEqClassIds(aliasAnalysis, inst.getInstOuts(), liveEqClassIds);
    collectEqClassIds(aliasAnalysis, inst.getInstIns(), liveEqClassIds);
    transferFunction(after, before, {}, liveEqClassIds);
    return success();
  }

  // Handle other operations.
  // These operations can only make existing equivalence classes live before.
  SmallVector<EqClassID> liveEqClassIds;
  collectEqClassIds(aliasAnalysis, op->getOperands(), liveEqClassIds);
  transferFunction(after, before, {}, liveEqClassIds);
  return success();
}

void LivenessAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const LivenessState &after,
                                          LivenessState *before) {
  DUMP_STATE_HELPER(g1, "with successor", PrintableBlock{successor});
  DUMP_STATE_HELPER(g2, "within block", PrintableBlock{block});
  DUMP_STATE_HELPER(g3, "program point", PrintableProgramPoint{point});
  if (handleTopPropagation(after, before))
    return;
  SmallVector<EqClassID> deadEqClassIds;
  collectEqClassIds(aliasAnalysis, successor->getArguments(), deadEqClassIds);
  transferFunction(after, before, std::move(deadEqClassIds), {});
}

void LivenessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const LivenessState &after, LivenessState *before) {
  DUMP_STATE_HELPER(g1, "call op",
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
  DUMP_STATE_HELPER(g1, "branch op",
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
