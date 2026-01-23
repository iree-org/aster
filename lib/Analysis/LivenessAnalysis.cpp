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
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
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
  const LiveSet *values = getLiveValues();
  assert(values && "Live values should be valid here");

  // Sort values by program order for deterministic output.
  SmallVector<Value> sortedValues(values->begin(), values->end());
  llvm::sort(sortedValues, compareValuesByProgramOrder);

  os << "[";
  llvm::interleaveComma(sortedValues, os, [&](Value value) {
    value.printAsOperand(os, OpPrintingFlags());
  });
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
    liveValues = *lattice.liveValues;
    return ChangeResult::Change;
  }
  const LiveSet *latticeVals = lattice.getLiveValues();
  assert(latticeVals && "Lattice values should be valid here");
  return appendValues(*latticeVals);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// Transfer function for liveness analysis.
void LivenessAnalysis::transferFunction(const LivenessState &after,
                                        LivenessState *before,
                                        SmallVector<Value> &&deadValues,
                                        ValueRange inValues) {
  auto cmpVal = +[](Value a, Value b) {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  };
  SmallVector<Value> liveValues;
  SmallVector<Value> afterValues = llvm::to_vector(*after.getLiveValues());
  if (!afterValues.empty())
    llvm::sort(afterValues, cmpVal);
  if (!deadValues.empty())
    llvm::sort(deadValues, cmpVal);
  std::set_difference(afterValues.begin(), afterValues.end(),
                      deadValues.begin(), deadValues.end(),
                      std::back_inserter(liveValues), cmpVal);
  llvm::append_range(liveValues, inValues);
  propagateIfChanged(before, before->appendValues(liveValues));
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

LogicalResult LivenessAnalysis::visitOperation(Operation *op,
                                               const LivenessState &after,
                                               LivenessState *before) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return success();

  // Handle instruction operations.
  if (auto inst = dyn_cast<InstOpInterface>(op)) {
    SmallVector<Value> deadValues = llvm::to_vector_of<Value>(op->getResults());
    // Append instruction outputs to dead values, as they are actually result
    // values.
    llvm::append_range(deadValues, inst.getInstOuts());
    transferFunction(after, before, std::move(deadValues), inst.getInstIns());
    return success();
  }

  // Handle MakeRegisterRangeOp.
  if (auto mOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
    SmallVector<Value> deadValues = llvm::to_vector_of<Value>(op->getResults());
    // The inputs don't participate in the transfer function as this op doesn't
    // change liveness.
    transferFunction(after, before, std::move(deadValues), {});
    return success();
  }

  // Handle SplitRegisterRangeOp.
  if (auto sOp = dyn_cast<amdgcn::SplitRegisterRangeOp>(op)) {
    SmallVector<Value> deadValues = llvm::to_vector_of<Value>(op->getResults());
    // The inputs don't participate in the transfer function as this op doesn't
    // change liveness.
    transferFunction(after, before, std::move(deadValues), op->getOperands());
    return success();
  }

  // Handle RegInterferenceOp.
  if (auto iOp = dyn_cast<amdgcn::RegInterferenceOp>(op)) {
    // Reg interference operations do not affect liveness.
    transferFunction(after, before, {}, {});
    return success();
  }

  // Handle generic operations.
  transferFunction(after, before, llvm::to_vector_of<Value>(op->getResults()),
                   op->getOperands());
  return success();
}

void LivenessAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const LivenessState &after,
                                          LivenessState *before) {
  DUMP_STATE_HELPER("block", block);
  if (handleTopPropagation(after, before))
    return;
  transferFunction(after, before,
                   llvm::to_vector_of<Value>(successor->getArguments()), {});
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
  transferFunction(after, before, llvm::to_vector_of<Value>(call->getResults()),
                   call.getArgOperands());
}

void LivenessAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionSuccessor regionTo, const LivenessState &after,
    LivenessState *before) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return;
  transferFunction(
      after, before,
      llvm::to_vector_of<Value>(branch.getSuccessorInputs(regionTo)), {});
}

void LivenessAnalysis::setToExitState(LivenessState *lattice) {
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

#undef DUMP_STATE_HELPER
