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
#include <cstdint>
#include <optional>

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// AliasEquivalenceClass
//===----------------------------------------------------------------------===//

using EqClassID = int32_t;

/// Lattice value representing equivalence class information.
class AliasEquivalenceClass {
public:
  using EqClassList = llvm::SmallVector<EqClassID, 4>;

  /// Construct a equivalence class value.
  AliasEquivalenceClass(EqClassList eqClassIds = {})
      : eqClassIds(std::move(eqClassIds)) {}

  /// Compare equivalence class values.
  bool operator==(const AliasEquivalenceClass &rhs) const {
    return succeeded(rhs.eqClassIds) == succeeded(eqClassIds) &&
           (failed(rhs.eqClassIds) || *rhs.eqClassIds == *eqClassIds);
  }

  /// Print the equivalence class value.
  void print(raw_ostream &os) const;

  /// The state to which the equivalence class value is uninitialized.
  static AliasEquivalenceClass getUninitialized() {
    return AliasEquivalenceClass{};
  }

  /// The state to which the equivalence class value is overdefined.
  static AliasEquivalenceClass getTop() {
    return AliasEquivalenceClass(std::nullopt);
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const {
    return llvm::succeeded(eqClassIds) && eqClassIds->empty();
  }

  /// Whether the state is top (overdefined).
  bool isTop() const { return llvm::failed(eqClassIds); }

  /// Returns TOP if IDs differ (ill-formed IR or bug in ValueProvenanceAnalysis
  /// where phi-coalesced allocas weren't unified).
  static AliasEquivalenceClass join(const AliasEquivalenceClass &lhs,
                                    const AliasEquivalenceClass &rhs) {
    if (lhs.isTop() || rhs.isTop())
      return getTop();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (*lhs.eqClassIds == *rhs.eqClassIds)
      return lhs;
    // Different IDs at join = ill-formed IR. See class documentation.
    return AliasEquivalenceClass(std::nullopt);
  }

  /// Get the equivalence class IDs. Returns an empty list if uninitialized or
  /// top.
  ArrayRef<EqClassID> getEqClassIds() const {
    return succeeded(eqClassIds) ? *eqClassIds : ArrayRef<EqClassID>();
  }

private:
  /// Construct an equivalence class value from a LogicalResult. This only works
  /// for failure states.
  explicit AliasEquivalenceClass(std::nullopt_t) : eqClassIds(failure()) {}
  // The equivalence class IDs.
  llvm::FailureOr<EqClassList> eqClassIds;
};

//===----------------------------------------------------------------------===//
// DPSAliasAnalysis
//===----------------------------------------------------------------------===//

/// DPS alias analysis assigns alias-equivalence class IDs to values for
/// register allocation interference analysis.
///
/// Runs after ValueProvenanceAnalysis to properly track phi-coalesced values.
///
/// TOP state indicates ill-formed IR: non-equivalent allocas merging at a
/// block argument.
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

  /// Whether the analysis detected ill-formed equivalence class usage.
  bool isIllFormedIR() const { return illFormed; }

  /// Get the underlying data flow solver.
  const DataFlowSolver &getSolver() const { return solver; }

  /// Lookup the equivalence class state for a given value.
  const AliasEquivalenceClass *lookupState(Value v) const {
    auto *state =
        solver.lookupState<dataflow::Lattice<AliasEquivalenceClass>>(v);
    return state ? &state->getValue() : nullptr;
  }

  /// Get the equivalence class IDs for a given value.
  ArrayRef<EqClassID> getEqClassIds(Value v) const {
    auto *state = lookupState(v);
    return state ? state->getEqClassIds() : ArrayRef<EqClassID>();
  }

  /// Get the values corresponding to equivalence class IDs.
  ArrayRef<Value> getValues() const { return idsToValuesMap; }

protected:
  DenseMap<Value, EqClassID> valueToEqClassIdMap;
  SmallVector<Value> idsToValuesMap;
  const DataFlowSolver &solver;
  bool illFormed = false;

private:
  // Optional provenance analysis for phi-coalescing. Null = no phi-coalescing.
  const ValueProvenanceAnalysis *provenanceAnalysis;

  // Equivalence classes resulting from DPS alias analysis.
  EqClassID getOrCreateEqClassId(Value alloca);
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_DPSALIASANALYSIS_H
