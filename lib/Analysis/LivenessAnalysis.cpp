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
// Helper functions
//===----------------------------------------------------------------------===//

/// Compare two values by program order for deterministic sorting.
/// Block arguments come before op results. Block arguments are compared by
/// block and argument number. Op results are compared by operation order.
static bool compareValuesByProgramOrder(Value a, Value b) {
  // Block arguments come before op results.
  bool aIsArg = isa<BlockArgument>(a);
  bool bIsArg = isa<BlockArgument>(b);
  if (aIsArg != bIsArg)
    return aIsArg;

  if (aIsArg) {
    // Both are block arguments - compare by block and arg number.
    auto argA = cast<BlockArgument>(a);
    auto argB = cast<BlockArgument>(b);
    if (argA.getOwner() != argB.getOwner())
      return argA.getOwner() < argB.getOwner();
    return argA.getArgNumber() < argB.getArgNumber();
  }

  // Both are op results - compare by operation order.
  Operation *opA = a.getDefiningOp();
  Operation *opB = b.getDefiningOp();
  if (opA == opB)
    return cast<OpResult>(a).getResultNumber() <
           cast<OpResult>(b).getResultNumber();
  if (opA->getBlock() == opB->getBlock())
    return opA->isBeforeInBlock(opB);
  // Different blocks - use pointer comparison for stability.
  return opA->getBlock() < opB->getBlock();
}

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
  const ValueSet &values = getLiveValues();
  const EqClassSet &eqClassIds = getLiveEqClassIds();

  // Sort values by program order for deterministic output.
  SmallVector<Value> sortedValues(values.begin(), values.end());
  llvm::sort(sortedValues, compareValuesByProgramOrder);

  // Sort eq class IDs for deterministic output.
  SmallVector<EqClassID> sortedEqClassIds(eqClassIds.begin(), eqClassIds.end());
  llvm::sort(sortedEqClassIds);

  os << "[values: ";
  llvm::interleaveComma(sortedValues, os, [&](Value value) {
    value.printAsOperand(os, OpPrintingFlags());
  });
  os << ", eqClasses: ";
  llvm::interleaveComma(sortedEqClassIds, os,
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
    isTopState = false;
    liveValues = lattice.liveValues;
    liveEqClasses = lattice.liveEqClasses;
    return ChangeResult::Change;
  }
  const ValueSet &latticeVals = lattice.getLiveValues();
  const EqClassSet &latticeEqClasses = lattice.getLiveEqClassIds();
  return updateLiveness(latticeVals, latticeEqClasses);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// Transfer function for liveness analysis.
/// - deadValues: Results being defined (dead going backwards)
/// - liveValues: Operands always live with their eq classes
/// - aliasingOperands: Operands live but eq classes conditional on callback
/// - isEqClassLive: Returns true if aliasing operand's eq class should be live
void LivenessAnalysis::transferFunction(
    const LivenessState &after, LivenessState *before,
    ArrayRef<Value> deadValues, ArrayRef<Value> liveValues,
    ArrayRef<Value> aliasingOperands, function_ref<bool(Value)> isEqClassLive) {
  // Start with values that are live after this point.
  LivenessState::ValueSet resultValues;
  if (!after.isTop())
    resultValues.insert_range(after.getLiveValues());

  // Remove dead values (definitions going backwards).
  for (Value v : deadValues)
    resultValues.erase(v);

  // Add all operands as live values.
  resultValues.insert_range(liveValues);
  resultValues.insert_range(aliasingOperands);

  // Propagate eq classes independently from values.
  LivenessState::EqClassSet resultEqClasses;
  if (!after.isTop())
    resultEqClasses.insert_range(after.getLiveEqClassIds());

  // Remove eq classes for dead values (definitions going backwards).
  for (Value v : deadValues) {
    if (!isa<RegisterTypeInterface>(v.getType()))
      continue;
    for (EqClassID id : aliasAnalysis->getEqClassIds(v))
      resultEqClasses.erase(id);
  }

  // Add eq classes for regular operands (always live).
  for (Value v : liveValues) {
    if (!isa<RegisterTypeInterface>(v.getType()))
      continue;
    resultEqClasses.insert_range(aliasAnalysis->getEqClassIds(v));
  }

  // Add eq classes for aliasing operands only if callback returns true.
  for (Value v : aliasingOperands) {
    if (!isa<RegisterTypeInterface>(v.getType()))
      continue;
    if (isEqClassLive(v))
      resultEqClasses.insert_range(aliasAnalysis->getEqClassIds(v));
  }

  propagateIfChanged(before,
                     before->updateLiveness(resultValues, resultEqClasses));
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

#define DUMP_STATE_HELPER(uniqueGuardName, name, obj, pathTaken)               \
  auto uniqueGuardName = llvm::make_scope_exit([&]() {                         \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "Visiting " name ": " << obj << "\n";                              \
      os << "  Incoming lattice: ";                                            \
      after.print(os);                                                         \
      os << "\n  Path taken: " << pathTaken;                                   \
      os << "\n  Outgoing lattice: ";                                          \
      before->print(os);                                                       \
    });                                                                        \
  });

/// Helper to collect values with register types from a range.
static void collectRegisterValues(ValueRange values,
                                  SmallVectorImpl<Value> &result) {
  for (Value v : values) {
    if (isa<RegisterTypeInterface>(v.getType()))
      result.push_back(v);
  }
}

LogicalResult LivenessAnalysis::visitOperation(Operation *op,
                                               const LivenessState &after,
                                               LivenessState *before) {
  int pathTaken = 0;
  DUMP_STATE_HELPER(g1, "op", OpWithFlags(op, OpPrintingFlags().skipRegions()),
                    pathTaken);
  if (handleTopPropagation(after, before)) {
    pathTaken = 1;
    return success();
  }
  assert(!after.isTop() &&
         "After state should not be top after top propagation");

  SmallVector<Value> deadValues, pureSSAOperands, aliasingSSAOperands;

  // AllocaOp defines a new value (and equivalence class).
  // Going backwards, the value is dead before.
  if (auto allocaOp = dyn_cast<amdgcn::AllocaOp>(op)) {
    collectRegisterValues(allocaOp.getResult(), deadValues);
    transferFunction(after, before, deadValues, {});
    pathTaken = 2;
    return success();
  }

  // Handle aliasing operations: MakeRegisterRangeOp, SplitRegisterRangeOp,
  // InstOpInterface. These define aliasing relationships between operands and
  // results:
  // - MakeRegisterRangeOp: 1-to-N (single result aliases all operands)
  // - SplitRegisterRangeOp: N-to-1 (single operand aliases all results)
  // - InstOpInterface: 1-to-1 (result[i] aliases outs[i])
  //
  // Value liveness: all operands always live (true SSA).
  // Eq class liveness: aliasing operand's eq class live iff any aliased result
  // was live.
  if (isa<amdgcn::MakeRegisterRangeOp, amdgcn::SplitRegisterRangeOp>(op) ||
      isa<InstOpInterface>(op)) {
    DenseMap<Value, SmallVector<Value>> operandToResults;

    collectRegisterValues(op->getResults(), deadValues);

    if (auto makeRangeOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
      // 1-to-N: single result aliases all operands.
      collectRegisterValues(makeRangeOp.getOperands(), aliasingSSAOperands);
      for (Value operand : aliasingSSAOperands)
        operandToResults[operand].push_back(makeRangeOp.getResult());
    } else if (auto splitRangeOp = dyn_cast<amdgcn::SplitRegisterRangeOp>(op)) {
      // N-to-1: single operand aliases all results.
      collectRegisterValues(splitRangeOp.getInput(), aliasingSSAOperands);
      for (Value operand : aliasingSSAOperands)
        for (Value result : splitRangeOp.getResults())
          operandToResults[operand].push_back(result);
    } else {
      // InstOpInterface: 1-to-1 (result[i] aliases outs[i]).
      auto inst = cast<InstOpInterface>(op);
      collectRegisterValues(inst.getInstIns(), pureSSAOperands);
      collectRegisterValues(inst.getInstOuts(), aliasingSSAOperands);
      for (auto [result, out] : llvm::zip(op->getResults(), inst.getInstOuts()))
        if (isa<RegisterTypeInterface>(out.getType()))
          operandToResults[out].push_back(result);
    }

    const LivenessState::ValueSet &liveAfter = after.getLiveValues();
    auto isEqClassLive = [&](Value operand) {
      auto it = operandToResults.find(operand);
      return it != operandToResults.end() &&
             llvm::any_of(it->second,
                          [&](Value r) { return liveAfter.contains(r); });
    };

    transferFunction(after, before, deadValues, pureSSAOperands,
                     aliasingSSAOperands, isEqClassLive);
    pathTaken = 3;
    return success();
  }

  // Handle RegInterferenceOp.
  if (isa<amdgcn::RegInterferenceOp>(op)) {
    // Reg interference operations do not affect liveness.
    transferFunction(after, before, {}, {});
    pathTaken = 4;
    return success();
  }

  // Handle other operations.
  // Results are defined here (dead going backwards).
  // Operands are used here (live going backwards).
  collectRegisterValues(op->getResults(), deadValues);
  SmallVector<Value> liveValues;
  collectRegisterValues(op->getOperands(), liveValues);
  transferFunction(after, before, deadValues, liveValues);
  pathTaken = 5;
  return success();
}

void LivenessAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const LivenessState &after,
                                          LivenessState *before) {
  int pathTaken = 0;
  DUMP_STATE_HELPER(g1, "with successor", PrintableBlock{successor}, pathTaken);
  DUMP_STATE_HELPER(g2, "within block", PrintableBlock{block}, pathTaken);
  DUMP_STATE_HELPER(g3, "program point", PrintableProgramPoint{point},
                    pathTaken);
  if (handleTopPropagation(after, before)) {
    pathTaken = 1;
    return;
  }
  SmallVector<Value> deadValues;
  collectRegisterValues(successor->getArguments(), deadValues);
  transferFunction(after, before, deadValues, {});
  pathTaken = 2;
}

void LivenessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const LivenessState &after, LivenessState *before) {
  int pathTaken = 0;
  DUMP_STATE_HELPER(g1, "call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()),
                    pathTaken);
  if (handleTopPropagation(after, before)) {
    pathTaken = 1;
    return;
  }
  assert(action == dataflow::CallControlFlowAction::ExternalCallee &&
         "we don't support inter-procedural analysis");
  SmallVector<Value> deadValues;
  collectRegisterValues(call->getResults(), deadValues);
  SmallVector<Value> liveValues;
  collectRegisterValues(call.getArgOperands(), liveValues);
  transferFunction(after, before, deadValues, liveValues);
  pathTaken = 2;
}

void LivenessAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionSuccessor regionTo, const LivenessState &after,
    LivenessState *before) {
  int pathTaken = 0;
  DUMP_STATE_HELPER(g1, "branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()),
                    pathTaken);
  if (handleTopPropagation(after, before)) {
    pathTaken = 1;
    return;
  }
  SmallVector<Value> deadValues;
  collectRegisterValues(llvm::to_vector_of<Value>(branch.getSuccessorInputs(regionTo)), deadValues);
  SmallVector<Value> liveValues;
  // The branch operation's operands (e.g., the condition in cf.cond_br) are
  // used by the branch and should be live before it.
  collectRegisterValues(branch->getOperands(), liveValues);
  transferFunction(after, before, deadValues, liveValues);
  pathTaken = 2;
}

void LivenessAnalysis::setToExitState(LivenessState *lattice) {
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

#undef DUMP_STATE_HELPER
