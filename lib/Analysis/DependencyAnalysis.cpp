//===- DependencyAnalysis.cpp - Dependency analysis -----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DependencyAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/DependentOpInterface.h"
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

#define DEBUG_TYPE "dependency-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// DependencyState
//===----------------------------------------------------------------------===//

void DependencyState::print(raw_ostream &os) const {
  if (isEmpty()) {
    os << "[]";
    return;
  }
  if (isTop()) {
    os << "<top>";
    return;
  }
  const DependencySet *tokens = getPendingTokens();
  assert(tokens && "Pending tokens should be valid here");
  os << "[";
  llvm::interleaveComma(*tokens, os, [&](Value value) {
    value.printAsOperand(os, OpPrintingFlags());
  });
  os << "]";
}

ChangeResult DependencyState::join(const DependencyState &lattice) {
  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  if (isTop())
    return ChangeResult::NoChange;

  if (lattice.isTop())
    return setToTop();

  if (isEmpty()) {
    pendingTokens = *lattice.pendingTokens;
    return ChangeResult::Change;
  }
  const DependencySet *latticeTokens = lattice.getPendingTokens();
  assert(latticeTokens && "Lattice tokens should be valid here");
  size_t oldSize = pendingTokens->size();
  pendingTokens->insert_range(*latticeTokens);
  return pendingTokens->size() != oldSize ? ChangeResult::Change
                                          : ChangeResult::NoChange;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::DependencyState)

//===----------------------------------------------------------------------===//
// DependencyAnalysis
//===----------------------------------------------------------------------===//

bool DependencyAnalysis::handleTopPropagation(const DependencyState &before,
                                              DependencyState *after) {
  if (before.isTop() || after->isTop()) {
    propagateIfChanged(after, after->setToTop());
    return true;
  }
  return false;
}

#define DUMP_STATE_HELPER(name, obj)                                           \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "Visiting " name ": " << obj << "\n";                              \
      os << "  Incoming lattice: ";                                            \
      before.print(os);                                                        \
      os << "\n  Outgoing lattice: ";                                          \
      after->print(os);                                                        \
    });                                                                        \
  });

LogicalResult DependencyAnalysis::visitOperation(Operation *op,
                                                 const DependencyState &before,
                                                 DependencyState *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(before, after))
    return success();

  // Start with the incoming state.
  ChangeResult changed = after->join(before);

  // Handle operations implementing DependentOpInterface: they produce and
  // consume tokens.
  if (auto depOp = dyn_cast<DependentOpInterface>(op)) {
    Value outToken = depOp.getOutDependency();
    if (outToken)
      changed |= after->addTokens({outToken});
    SmallVector<Value> consumedTokens =
        llvm::to_vector(depOp.getDependencies());
    changed |= after->removeTokens(consumedTokens);
    propagateIfChanged(after, changed);
    return success();
  }

  // For other operations, just propagate the state.
  propagateIfChanged(after, changed);
  return success();
}

void DependencyAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                            Block *predecessor,
                                            const DependencyState &before,
                                            DependencyState *after) {
  DUMP_STATE_HELPER("block", block);
  if (handleTopPropagation(before, after))
    return;
  propagateIfChanged(after, after->join(before));
}

void DependencyAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const DependencyState &before, DependencyState *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(before, after))
    return;
  assert(action == dataflow::CallControlFlowAction::ExternalCallee &&
         "we don't support inter-procedural analysis");
  // For external calls, we conservatively propagate the state.
  propagateIfChanged(after, after->join(before));
}

void DependencyAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const DependencyState &before,
    DependencyState *after) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(before, after))
    return;
  propagateIfChanged(after, after->join(before));
}

void DependencyAnalysis::setToEntryState(DependencyState *lattice) {
  // Entry state is empty (no pending tokens at function entry).
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

#undef DUMP_STATE_HELPER
