//===- HazardAnalysis.cpp - Hazard dataflow analysis ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/HazardAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "hazard-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// HazardState
//===----------------------------------------------------------------------===//

/// Merge two sorted vectors of hazards using merge sort. When two equivalent
/// hazards are found, use Hazard::join to combine them.
static bool mergeActiveHazards(SmallVectorImpl<Hazard> &merged,
                               ArrayRef<Hazard> lhs, ArrayRef<Hazard> rhs) {
  merged.reserve(lhs.size() + rhs.size());
  bool changed = false;

  int64_t i = 0, j = 0, lhsSize = lhs.size(), rhsSize = rhs.size();
  while (i < lhsSize && j < rhsSize) {
    if (lhs[i] < rhs[j]) {
      assert(lhs[i].isValid() && "lhs hazard should be active");
      merged.push_back(lhs[i]);
      ++i;
    } else if (rhs[j] < lhs[i]) {
      assert(rhs[j].isValid() && "rhs hazard should be active");
      merged.push_back(rhs[j]);
      ++j;
    } else {
      Hazard joined = lhs[i].join(rhs[j]);
      assert(joined.isValid() && "joined hazard should be active");
      if (joined != lhs[i])
        changed = true;
      merged.push_back(joined);
      ++i;
      ++j;
    }
  }
  while (i < lhsSize) {
    assert(lhs[i].isValid() && "lhs hazard should be active");
    merged.push_back(lhs[i]);
    ++i;
  }
  while (j < rhsSize) {
    assert(rhs[j].isValid() && "rhs hazard should be active");
    merged.push_back(rhs[j]);
    ++j;
  }

  // We always merge the rhs into the lhs, so if sizes are different, there was
  // a change.
  if (merged.size() != lhs.size())
    changed = true;
  return changed;
}

ChangeResult HazardState::join(const HazardState &lattice) {
  // If the rhs is empty, no change.
  if (lattice.activeHazards.empty())
    return ChangeResult::NoChange;

  // If the lhs is empty, just append the rhs.
  if (activeHazards.empty()) {
    llvm::append_range(activeHazards, lattice.activeHazards);
    return ChangeResult::Change;
  }

  // Add the active hazards from the rhs.
  return addHazards(lattice.activeHazards);
}

ChangeResult HazardState::addHazards(ArrayRef<Hazard> hazards) {
  if (hazards.empty())
    return ChangeResult::NoChange;

  ChangeResult changed = ChangeResult::NoChange;

  // Merge the hazards into the active hazards.
  SmallVector<Hazard, 4> merged;
  if (mergeActiveHazards(merged, activeHazards, hazards))
    changed = ChangeResult::Change;

  if (changed == ChangeResult::NoChange)
    return changed;

  activeHazards = std::move(merged);
  return changed;
}

/// Helper function to print a hazard state.
static void printHazardState(raw_ostream &os, ArrayRef<Hazard> hazards,
                             InstCounts counts) {
  os << "{\n  active = [";
  if (!hazards.empty()) {
    os << "\n";
    llvm::interleave(
        hazards, os, [&](const Hazard &h) { os << "    " << h; }, ",\n");
    os << "\n  ]\n";
  } else {
    os << "]\n";
  }
  os << "  nop counts = " << counts << "\n}";
}

void HazardState::print(raw_ostream &os) const {
  if (isEmpty()) {
    os << "<Empty>";
    return;
  }

  printHazardState(os, activeHazards, flushedInstCounts);
}

void HazardState::printDeterministic(raw_ostream &os,
                                     DominanceInfo &domInfo) const {
  if (isEmpty()) {
    os << "<Empty>";
    return;
  }

  // Sort the active hazards for deterministic output.
  SmallVector<Hazard> sortedActive(activeHazards.begin(), activeHazards.end());
  llvm::sort(sortedActive, [&](const Hazard &a, const Hazard &b) {
    return a.compare(b, domInfo);
  });

  printHazardState(os, sortedActive, flushedInstCounts);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::HazardState)

//===----------------------------------------------------------------------===//
// HazardAnalysis
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

/// Update the hazards based on the instruction counts that are being flushed.
/// Returns true if the hazards have changed.
static bool updateHazards(SmallVectorImpl<Hazard> &hazards,
                          InstCounts instCounts, AMDGCNInstOpInterface instOp) {
  int8_t numVector = instCounts.getNumVector();
  int8_t numScalar = instCounts.getNumScalar();
  int8_t numDataShare = instCounts.getNumDataShare();

  // Update the instruction counts based on the instruction interface.
  if (instOp) {
    if (instOp.hasAnyProps({InstProp::IsValu, InstProp::IsVmem}))
      ++numVector;
    if (instOp.hasAnyProps({InstProp::Salu, InstProp::Smem}))
      ++numScalar;
    // NOTE: It's unclear whether dsmem also affect vector instructions.
    if (instOp.hasAnyProps({InstProp::Dsmem}))
      ++numDataShare;
  }

  // Create a new instruction counts object with the updated counts.
  instCounts = InstCounts(numVector, numScalar, numDataShare);

  // Track if the hazards have changed.
  bool changed = false;

  // Decrement the instruction counts for each hazard and clear if inactive.
  // NOTE: This is critical to avoid introducing extra nop instructions.
  for (Hazard &h : hazards) {
    InstCounts hInstCounts = h.getInstCounts();
    h.getInstCounts().decrementCount(instCounts);
    h.clearIfInactive();

    // If the hazard has changed, set the changed flag.
    if (!(hInstCounts == h.getInstCounts()) || !h.isValid())
      changed = true;
  }

  // Remove the inactive hazards, this keeps the active hazards sorted.
  hazards.erase(
      llvm::remove_if(hazards, [](const Hazard &h) { return !h.isValid(); }),
      hazards.end());
  return changed;
}

LogicalResult HazardAnalysis::visitOperation(Operation *op,
                                             const HazardState &before,
                                             HazardState *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()), {});

  // Join the before and after states.
  ChangeResult changed = after->join(before);

  auto instOp = dyn_cast<amdgcn::AMDGCNInstOpInterface>(op);

  // Ignore non-instruction operations.
  if (!instOp) {
    LDBG() << "Skipping non-instruction operation: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());
    propagateIfChanged(after, changed);
    return success();
  }

  // Check if any of the active hazards are triggered by the instruction.
  InstCounts requiredInstCounts;
  for (Hazard &h : after->activeHazards) {
    // If the hazard is triggered by the instruction, clear the hazard and add
    // the instruction counts to the required instruction counts.
    if (h.getHazard().isHazardTriggered(h, instOp)) {
      // Add the instruction counts to the required instruction counts.
      requiredInstCounts.joinWithMax(h.getInstCounts());
      h.clear();
    }

    // If the hazard is inactive, the state has changed.
    if (!h.isActive())
      changed = ChangeResult::Change;
  }

  // Update the hazards based on the instruction counts that are being flushed.
  if (updateHazards(after->activeHazards, requiredInstCounts, instOp))
    changed = ChangeResult::Change;

  // Update the flushed instruction counts if they have changed.
  if (!(requiredInstCounts == after->flushedInstCounts)) {
    changed = ChangeResult::Change;
    after->flushedInstCounts = requiredInstCounts;
  }

  SmallVector<Hazard, 4> hazards;
  // Add the new hazards.
  if (failed(hazardManager.getHazards(instOp, hazards)))
    return instOp.emitError() << "failed to get hazards for instruction";

  // Remove the inactive hazards.
  hazards.erase(
      llvm::remove_if(hazards, [](const Hazard &h) { return !h.isActive(); }),
      hazards.end());

  // Sort the hazards for the merge sort.
  llvm::sort(hazards);

  changed |= after->addHazards(hazards);

  propagateIfChanged(after, changed);
  return success();
}

void HazardAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                        Block *predecessor,
                                        const HazardState &before,
                                        HazardState *after) {
  DUMP_STATE_HELPER("block", block, {});
  propagateIfChanged(after, after->join(before));
}

void HazardAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const HazardState &before, HazardState *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()), {});
  assert(false && "inter-procedural hazard analysis not supported");
}

void HazardAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const HazardState &before,
    HazardState *after) {
  DUMP_STATE_HELPER(
      "branch op", OpWithFlags(branch, OpPrintingFlags().skipRegions()), {
        os << "\n  Branching from: " << (regionFrom ? *regionFrom : -1)
           << " to " << (regionTo ? *regionTo : -1);
      });
  propagateIfChanged(after, after->join(before));
}

void HazardAnalysis::setToEntryState(HazardState *lattice) {
  bool changed = !lattice->activeHazards.empty() ||
                 !lattice->flushedInstCounts.isInactive();
  lattice->activeHazards.clear();
  lattice->flushedInstCounts.clear();
  propagateIfChanged(lattice,
                     changed ? ChangeResult::Change : ChangeResult::NoChange);
}

#undef DUMP_STATE_HELPER
