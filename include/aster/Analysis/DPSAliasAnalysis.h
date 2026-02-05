//===- DPSAliasAnalysis.h - DPS alias analysis ----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements DPS alias analysis using sparse data-flow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_DPSALIASANALYSIS_H
#define ASTER_ANALYSIS_DPSALIASANALYSIS_H

#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// AliasEquivalenceClass
//===----------------------------------------------------------------------===//

using EqClassID = int32_t;

/// Lattice value representing equivalence class information.
///
/// States (ordered from bottom to top):
///   UNINITIALIZED - No information yet, waiting for dataflow
///   KNOWN         - Concrete equivalence class ID(s)
///   UNKNOWN       - Cannot determine alias class (conservative)
///   CONFLICT      - Different allocas merged at join (will trip register
///                   allocation when used in that context)
class AliasEquivalenceClass {
public:
  using EqClassList = llvm::SmallVector<EqClassID, 4>;

  /// Internal state enum for clarity.
  enum class State : char {
    Uninitialized, // Bottom: no info yet
    Known,         // Has concrete eq class IDs
    Unknown,       // Cannot determine, but not an error
    Conflict       // Different allocas merged = DPS violation
  };

  /// Construct a known equivalence class value.
  AliasEquivalenceClass(EqClassList eqClassIds = {})
      : eqClassIds(std::move(eqClassIds)), state(State::Uninitialized) {
    if (!this->eqClassIds.empty())
      state = State::Known;
  }

  /// Compare equivalence class values.
  bool operator==(const AliasEquivalenceClass &rhs) const {
    if (state != rhs.state)
      return false;
    if (state == State::Known)
      return eqClassIds == rhs.eqClassIds;
    return true; // Non-Known states with same state are equal
  }

  /// Print the equivalence class value.
  void print(raw_ostream &os) const;

  /// The state to which the equivalence class value is uninitialized.
  static AliasEquivalenceClass getUninitialized() {
    return AliasEquivalenceClass{};
  }

  /// Unknown alias class (conservative).
  static AliasEquivalenceClass getUnknown() {
    AliasEquivalenceClass result;
    result.state = State::Unknown;
    return result;
  }

  /// Conflicting alias information (will trip register allocation).
  static AliasEquivalenceClass getConflict() {
    AliasEquivalenceClass result;
    result.state = State::Conflict;
    return result;
  }

  /// Legacy alias for getConflict() - will be removed.
  // RVW: drop this and reuse getConflict() instead everywhere.
  static AliasEquivalenceClass getTop() { return getConflict(); }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return state == State::Uninitialized; }

  /// Whether the state is known (has concrete eq class IDs).
  bool isKnown() const { return state == State::Known; }

  /// Whether the state is unknown (conservative, not error).
  bool isUnknown() const { return state == State::Unknown; }

  /// Whether the state is conflict (DPS violation, error).
  bool isConflict() const { return state == State::Conflict; }

  /// Legacy alias for isConflict() - will be removed.
  bool isTop() const { return isConflict(); }

  /// Join two lattice values. Returns CONFLICT only when different allocas
  /// merge.
  static AliasEquivalenceClass join(const AliasEquivalenceClass &lhs,
                                    const AliasEquivalenceClass &rhs) {
    // Conflict is absorbing (sticky error).
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();

    // Unknown is conservative: Unknown union X = Unknown (except Conflict).
    if (lhs.isUnknown() || rhs.isUnknown())
      return getUnknown();

    // Uninitialized is bottom: bottom union X = X.
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    // Both Known: same IDs = stable, different IDs = CONFLICT.
    if (lhs.eqClassIds == rhs.eqClassIds)
      return lhs;
    return getConflict();
  }

  /// Get the equivalence class IDs. Returns empty if not Known state.
  ArrayRef<EqClassID> getEqClassIds() const {
    return isKnown() ? ArrayRef<EqClassID>(eqClassIds) : ArrayRef<EqClassID>();
  }

  /// Get the current state.
  State getState() const { return state; }

private:
  EqClassList eqClassIds;
  State state = State::Uninitialized;
};

//===----------------------------------------------------------------------===//
// DPSAliasAnalysis
//===----------------------------------------------------------------------===//

/// DPS alias analysis assigns alias-equivalence class IDs to values for
/// register allocation interference analysis.
///
/// Runs after ValueProvenanceAnalysis to properly track phi-coalesced values.
///
/// CONFLICT state indicates IR that will trip register allocation:
/// non-equivalent allocas merging at a block argument.
/// UNKNOWN state is conservative.
class DPSAliasAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                             dataflow::Lattice<AliasEquivalenceClass>> {
public:
  DPSAliasAnalysis(DataFlowSolver &solver,
                   const ValueProvenanceAnalysis *provenance = nullptr)
      : SparseForwardDataFlowAnalysis(solver), solver(solver),
        provenanceAnalysis(provenance) {}

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::Lattice<AliasEquivalenceClass> *> operands,
      ArrayRef<dataflow::Lattice<AliasEquivalenceClass> *> results) override;

  void
  setToEntryState(dataflow::Lattice<AliasEquivalenceClass> *lattice) override;

  /// Lookup eq class ID for a value. Returns -1 if not found.
  EqClassID lookup(Value val) const {
    return valueToEqClassIdMap.lookup_or(val, -1);
  }

  /// Lookup the value assigned to the given equivalence class ID. Returns null
  /// Value if not found.
  Value lookup(EqClassID eqClassId) const {
    return static_cast<size_t>(eqClassId) < idsToValuesMap.size()
               ? idsToValuesMap[eqClassId]
               : Value();
  }

  /// Get values with CONFLICT state (will trip register allocation).
  ArrayRef<Value> getConflictingValues() const { return conflictingValues; }

  /// Get values with UNKNOWN state.
  ArrayRef<Value> getUnknownValues() const { return unknownValues; }

  /// Legacy alias for getConflictingValues() - will be removed.
  // RVW: drop this and reuse getConflictingValues() instead everywhere.
  ArrayRef<Value> getTopValues() const { return getConflictingValues(); }

  /// Get the underlying data flow solver.
  const DataFlowSolver &getSolver() const { return solver; }

  /// Lookup the equivalence class state for a given value.
  const AliasEquivalenceClass *lookupState(Value v) const {
    auto *state =
        solver.lookupState<dataflow::Lattice<AliasEquivalenceClass>>(v);
    return state ? &state->getValue() : nullptr;
  }

  /// Get the equivalence class IDs for a given value.
  /// Returns empty if value doesn't have Known state.
  ArrayRef<EqClassID> getEqClassIds(Value v) const {
    auto *state = lookupState(v);
    return (state && state->isKnown()) ? state->getEqClassIds()
                                       : ArrayRef<EqClassID>();
  }

  /// Get the values corresponding to equivalence class IDs.
  ArrayRef<Value> getValues() const { return idsToValuesMap; }

protected:
  // Map from values to equivalence class IDs.
  DenseMap<Value, EqClassID> valueToEqClassIdMap;
  // Map from equivalence class IDs to values.
  SmallVector<Value> idsToValuesMap;
  // Values with CONFLICT state (actual DPS violations).
  SmallVector<Value> conflictingValues;
  // Values with UNKNOWN state (conservative, not errors).
  SmallVector<Value> unknownValues;
  const DataFlowSolver &solver;

private:
  // Optional provenance analysis for phi-coalescing. Null = no phi-coalescing.
  const ValueProvenanceAnalysis *provenanceAnalysis;

  // Equivalence classes resulting from DPS alias analysis.
  EqClassID getOrCreateEqClassId(Value alloca);
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_DPSALIASANALYSIS_H
