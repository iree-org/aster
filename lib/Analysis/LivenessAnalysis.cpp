//===- LivenessAnalysis.cpp - Liveness analysis -----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "aster/Interfaces/LivenessOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "liveness-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// LivenessState
//===----------------------------------------------------------------------===//

void LivenessState::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<top>";
    return;
  }
  if (isEmpty()) {
    os << "[]";
    return;
  }
  const ValueSet *values = getLiveValues();

  os << "[";
  llvm::interleaveComma(*values, os, [&](Value value) {
    value.printAsOperand(os, OpPrintingFlags());
  });
  os << "]";
}

void LivenessState::print(raw_ostream &os, const SSAMap &ssaMap) const {
  if (isTop()) {
    os << "<top>";
    return;
  }
  if (isEmpty()) {
    os << "[]";
    return;
  }
  const ValueSet *values = getLiveValues();
  // Sort the ids to make the output deterministic.
  SmallVector<std::pair<Value, int64_t>> ids;
  ssaMap.getIds(*values, ids);
  llvm::sort(ids, [](const std::pair<Value, int64_t> &lhs,
                     const std::pair<Value, int64_t> &rhs) {
    return lhs.second < rhs.second;
  });
  os << "[";
  llvm::interleaveComma(ids, os, [&](const std::pair<Value, int64_t> &entry) {
    os << entry.second << " = `" << ValueWithFlags(entry.first, true) << "`";
  });
  os << "]";
}

ChangeResult LivenessState::meet(const LivenessState &lattice) {
  // Empty lattice contributes nothing.
  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  // Top absorbs everything.
  if (isTop())
    return ChangeResult::NoChange;

  // Meet with top results in top.
  if (lattice.isTop())
    return setToTop();

  // Both have concrete values - compute union.
  const ValueSet *otherValues = lattice.getLiveValues();

  // Initialize if we have no values yet.
  ChangeResult changed = ChangeResult::NoChange;

  // Compute union.
  size_t oldSize = liveValues->size();
  liveValues->insert(otherValues->begin(), otherValues->end());
  if (liveValues->size() != oldSize)
    changed = ChangeResult::Change;

  return changed;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

#define DUMP_STATE_HELPER(name, obj)                                           \
  LDBG_OS([&](raw_ostream &os) {                                               \
    os << "Visiting " name ": " << obj << "\n";                                \
    os << "  Incoming lattice: ";                                              \
    after.print(os);                                                           \
  });                                                                          \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "  Outgoing lattice: ";                                            \
      before->print(os);                                                       \
    });                                                                        \
  });

static bool hasRegisterType(Value value) {
  return isa<RegisterTypeInterface>(value.getType());
}

void LivenessAnalysis::transferFunction(const LivenessState &after,
                                        LivenessState *before,
                                        ValueRange deadValues,
                                        ValueRange inValues) {
  // Check if it's in the top state.
  LivenessState::ValueSet *livenessSet = before->getLiveValues();
  if (!livenessSet)
    return;

  // If there are no dead or in values, we can just meet with the after state.
  if (deadValues.empty() && inValues.empty()) {
    propagateIfChanged(before, before->meet(after));
    return;
  }

  // Meet with the after state.
  ChangeResult changed = before->meet(after);

  // Remove the dead values.
  for (Value deadValue : llvm::make_filter_range(deadValues, hasRegisterType)) {
    changed |= livenessSet->erase(deadValue) ? ChangeResult::Change
                                             : ChangeResult::NoChange;
  }

  // Add the in values.
  int64_t size = livenessSet->size();
  livenessSet->insert_range(llvm::make_filter_range(inValues, hasRegisterType));
  if (livenessSet->size() != size)
    changed = ChangeResult::Change;
  propagateIfChanged(before, changed);
}

LogicalResult
LivenessAnalysis::transferFunction(const LivenessState &after,
                                   LivenessState *before,
                                   LivenessOpInterface livenessOp) {
  // Check if it's in the top state.
  LivenessState::ValueSet *livenessSet = before->getLiveValues();
  if (!livenessSet)
    return success();

  // Meet with the after state.
  ChangeResult changed = before->meet(after);

  // Remove the dead values.
  auto addDead = [&](ValueRange deadValues) {
    for (Value deadValue :
         llvm::make_filter_range(deadValues, hasRegisterType)) {
      changed |= livenessSet->erase(deadValue) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
    }
  };

  // Add the live values.
  auto addLive = [&](ValueRange liveValues) {
    int64_t size = livenessSet->size();
    livenessSet->insert_range(
        llvm::make_filter_range(liveValues, hasRegisterType));
    if (livenessSet->size() != size)
      changed = ChangeResult::Change;
  };

  // Check if a value is live.
  auto isLive = [&](Value value) { return livenessSet->contains(value); };

  if (failed(livenessOp.livenessTransferFunction(addLive, addDead, isLive)))
    return failure();

  // Propagate the change.
  propagateIfChanged(before, changed);
  return success();
}

bool LivenessAnalysis::handleTopPropagation(const LivenessState &after,
                                            LivenessState *before) {
  if (after.isTop() || before->isTop()) {
    propagateIfChanged(before, before->setToTop());
    return true;
  }
  return false;
}

LogicalResult LivenessAnalysis::visitOperation(Operation *op,
                                               const LivenessState &after,
                                               LivenessState *before) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));

  // Handle top propagation.
  if (handleTopPropagation(after, before))
    return success();

  // Handle operations with LivenessOpInterface.
  if (auto livenessOp = dyn_cast<LivenessOpInterface>(op))
    return transferFunction(after, before, livenessOp);

  // Transfer function for operations without LivenessOpInterface.
  transferFunction(after, before, op->getResults(), op->getOperands());
  return success();
}

void LivenessAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const LivenessState &after,
                                          LivenessState *before) {
  DUMP_STATE_HELPER("block", block);

  // Handle top propagation.
  if (handleTopPropagation(after, before))
    return;

  // Kill the block arguments. Blocks themselves don't produce live values.
  transferFunction(after, before, successor->getArguments(), {});
}

void LivenessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const LivenessState &after, LivenessState *before) {
  DUMP_STATE_HELPER("call", call);

  // Handle top propagation.
  if (handleTopPropagation(after, before))
    return;

  // Kill the call results and use the call arguments as live values.
  transferFunction(after, before, call->getResults(), call.getArgOperands());
}

void LivenessAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionSuccessor regionTo, const LivenessState &after,
    LivenessState *before) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));

  // Handle top propagation.
  if (handleTopPropagation(after, before))
    return;

  // Kill the region inputs and use the region operands as live values.
  ValueRange inputs = branch.getSuccessorInputs(regionTo);
  ValueRange operands = branch.getSuccessorOperands(regionFrom, regionTo);
  transferFunction(after, before, inputs, operands);
}

void LivenessAnalysis::setToExitState(LivenessState *lattice) {
  // At exit points nothing is live initially.
  propagateIfChanged(lattice, lattice->setToEmpty());
}
