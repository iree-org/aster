//===- BufferAnalysis.cpp - Buffer analysis -------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/BufferAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "lds-buffer-analysis"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// BufferState
//===----------------------------------------------------------------------===//

/// Helper function to stringify a buffer state.
static StringRef stringifyState(BufferState::State state) {
  switch (state) {
  case BufferState::State::None:
    return "none";
  case BufferState::State::Live:
    return "live";
  case BufferState::State::Dead:
    return "dead";
  case BufferState::State::Top:
    return "top";
  }
  llvm_unreachable("unknown state");
}

void BufferState::print(raw_ostream &os) const {
  if (isEmpty()) {
    os << "[]";
    return;
  }
  os << "[";
  llvm::interleaveComma(buffers, os, [&](const std::pair<Value, State> &entry) {
    os << "{";
    entry.first.printAsOperand(os, OpPrintingFlags());
    os << ", " << stringifyState(entry.second) << "}";
  });
  os << "]";
}

/// Update the state of a buffer and return whether it changed.
static ChangeResult updateState(BufferState::State &state,
                                BufferState::State incomingState) {
  if (state == incomingState || state == BufferState::State::Top)
    return ChangeResult::NoChange;

  assert(incomingState != BufferState::State::None && "invalid incoming state");

  // If the buffer is not present, set it to the incoming state.
  if (state == BufferState::State::None) {
    state = incomingState;
    return ChangeResult::Change;
  }

  // At this point the states conflict, set to Top.
  state = BufferState::State::Top;
  return ChangeResult::Change;
}

ChangeResult BufferState::join(const BufferState &lattice,
                               llvm::function_ref<bool(Value)> dominates) {
  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  // If this state is empty, take all incoming buffers.
  if (isEmpty() && !dominates) {
    buffers = lattice.buffers;
    return ChangeResult::Change;
  }

  ChangeResult changed = ChangeResult::NoChange;

  // Merge the buffer states.
  for (auto [buffer, incomingState] : lattice.buffers) {
    // If a dominance function is provided, skip buffers that do not dominate.
    if (dominates && !dominates(buffer))
      continue;

    changed |= updateState(buffers[buffer], incomingState);
  }
  return changed;
}

ChangeResult BufferState::updateBuffer(Value buffer, State state) {
  return updateState(buffers[buffer], state);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::BufferState)

//===----------------------------------------------------------------------===//
// BufferAnalysis
//===----------------------------------------------------------------------===//

#define DUMP_STATE_HELPER(name, obj, extra)                                    \
  LDBG_OS([&](raw_ostream &os) {                                               \
    os << "Visiting " name ": " << obj << "\n";                                \
    os << "  Incoming lattice: ";                                              \
    before.print(os);                                                          \
    extra                                                                      \
  });                                                                          \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "  Outgoing lattice: ";                                            \
      after->print(os);                                                        \
    });                                                                        \
  });

void BufferAnalysis::setToEntryState(BufferState *lattice) {
  propagateIfChanged(lattice, lattice->markAllAsTop());
}

LogicalResult BufferAnalysis::visitOperation(Operation *op,
                                             const BufferState &before,
                                             BufferState *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()), {});
  // Start with the state before this operation.
  ChangeResult changed = after->join(before);

  if (auto dOp = dyn_cast<DeallocLDSOp>(op)) {
    changed |= after->killBuffer(dOp.getBuffer());
    propagateIfChanged(after, changed);
    return success();
  }

  if (auto aOp = dyn_cast<AllocLDSOp>(op)) {
    changed |= after->addBuffer(aOp.getBuffer());
    propagateIfChanged(after, changed);
    return success();
  }

  // For all other operations, just propagate the state unchanged.
  propagateIfChanged(after, changed);
  return success();
}

static ChangeResult
mapControlFlowOperands(BufferState &after, const BufferState &before,
                       ValueRange successorOperands, ValueRange successorValues,
                       llvm::function_ref<bool(Value)> dominates) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto operandValue :
       llvm::zip_equal(successorOperands, successorValues)) {
    Value operand = std::get<0>(operandValue);
    Value value = std::get<1>(operandValue);

    LDBG_OS([&](llvm::raw_ostream &os) {
      os << "  Checking propagated value from: ";
      operand.printAsOperand(os, OpPrintingFlags());
      os << " to ";
      value.printAsOperand(os, OpPrintingFlags());
    });

    BufferState::State state = before.getBufferState(operand);

    if (state == BufferState::State::None)
      continue;

    // Mark as dead if the operand does not dominate. This scopes the lifetime
    // of the buffer to the dominance frontier.
    if (!dominates(operand))
      state = BufferState::State::Dead;

    changed |= after.updateBuffer(value, state);
  }
  return changed;
}

void BufferAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                        Block *predecessor,
                                        const BufferState &before,
                                        BufferState *after) {
  DUMP_STATE_HELPER("block", block, {});

  auto dominates = [&](Value v) {
    return dominatesSuccessor(domInfo, v, block);
  };
  // Join the state from the predecessor using dominance information.
  ChangeResult changed = after->join(before, dominates);

  // Propagate tokens from the predecessor to this block.
  auto terminator = cast<BranchOpInterface>(predecessor->getTerminator());
  for (auto [i, succ] : llvm::enumerate(terminator->getSuccessors())) {
    if (succ != block)
      continue;
    changed |= mapControlFlowOperands(
        *after, before,
        terminator.getSuccessorOperands(i).getForwardedOperands(),
        block->getArguments(), dominates);
  }
  propagateIfChanged(after, changed);
}

void BufferAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const BufferState &before,
    BufferState *after) {
  DUMP_STATE_HELPER(
      "branch op", OpWithFlags(branch, OpPrintingFlags().skipRegions()), {
        os << "\n  Branching from: " << (regionFrom ? *regionFrom : -1)
           << " to " << (regionTo ? *regionTo : -1);
      });
  // Determine the successor.
  RegionSuccessor successor =
      regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
               : RegionSuccessor::parent();

  auto dominates = [&](Value v) {
    return dominatesSuccessor(domInfo, v, branch, successor);
  };
  // Join the states that are control-flow independent.
  ChangeResult changed = after->join(before, dominates);

  // Branch from parent.
  if (!regionFrom) {
    changed |= mapControlFlowOperands(
        *after, before,
        branch.getSuccessorOperands(RegionBranchPoint::parent(), successor),
        branch.getSuccessorInputs(successor), dominates);
  } else {
    // Branch from a region.
    walkTerminators(&branch->getRegion(*regionFrom),
                    [&](RegionBranchTerminatorOpInterface terminator) {
                      changed |= mapControlFlowOperands(
                          *after, before,
                          branch.getSuccessorOperands(
                              RegionBranchPoint(terminator), successor),
                          branch.getSuccessorInputs(successor), dominates);
                    });
  }
  propagateIfChanged(after, changed);
}
