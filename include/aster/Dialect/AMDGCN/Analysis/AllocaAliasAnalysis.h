//===- AllocaAliasAnalysis.h - Alloca alias analysis ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements alloca alias analysis using sparse data-flow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_ALLOCAALIASANALYSIS_H
#define ASTER_ANALYSIS_ALLOCAALIASANALYSIS_H

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
// AllocaAlias
//===----------------------------------------------------------------------===//

using AllocaID = int32_t;
/// This lattice value represents alloca alias information.
class AllocaAlias {
public:
  using IdList = llvm::SmallVector<AllocaID, 4>;
  /// Construct an alloca alias value.
  AllocaAlias(IdList allocaIds = {}) : allocaIds(std::move(allocaIds)) {}

  /// Compare alloca alias values.
  bool operator==(const AllocaAlias &rhs) const {
    return succeeded(rhs.allocaIds) == succeeded(allocaIds) &&
           (failed(rhs.allocaIds) || *rhs.allocaIds == *allocaIds);
  }

  /// Print the alloca alias value.
  void print(raw_ostream &os) const;

  /// The state where the alloca alias value is uninitialized.
  static AllocaAlias getUninitialized() { return AllocaAlias{}; }

  /// The state where the alloca alias value is overdefined.
  static AllocaAlias getTop() { return AllocaAlias(std::nullopt); }

  /// Whether the state is uninitialized.
  bool isUninitialized() const {
    return llvm::succeeded(allocaIds) && allocaIds->empty();
  }

  /// Whether the state is top (overdefined).
  bool isTop() const { return llvm::failed(allocaIds); }

  /// Join operation for lattice.
  static AllocaAlias join(const AllocaAlias &lhs, const AllocaAlias &rhs) {
    if (lhs.isTop() || rhs.isTop())
      return getTop();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (*lhs.allocaIds == *rhs.allocaIds)
      return lhs;
    /// If the alloca IDs differ, we conservatively return top.
    return AllocaAlias(std::nullopt);
  }

  /// Get the alloca IDs. Returns an empty list if uninitialized or top.
  ArrayRef<AllocaID> getAllocaIds() const {
    return succeeded(allocaIds) ? *allocaIds : ArrayRef<AllocaID>();
  }

private:
  /// Construct an alloca alias value from a LogicalResult. This only works for
  /// failure states.
  explicit AllocaAlias(std::nullopt_t) : allocaIds(failure()) {}
  llvm::FailureOr<IdList> allocaIds;
};

//===----------------------------------------------------------------------===//
// AllocaAliasAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements alloca alias analysis using sparse forward
/// data-flow analysis.
class AllocaAliasAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                                dataflow::Lattice<AllocaAlias>> {
public:
  AllocaAliasAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver), solver(solver) {}
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AllocaAlias> *> operands,
                 ArrayRef<dataflow::Lattice<AllocaAlias> *> results) override;

  void setToEntryState(dataflow::Lattice<AllocaAlias> *lattice) override;

  /// Lookup the alloca ID assigned to the given value. Returns -1 if not
  /// found.
  AllocaID lookup(Value val) const {
    return valueToAllocaIdMap.lookup_or(val, -1);
  }

  /// Lookup the value assigned to the given alloca ID. Returns null Value if
  /// not found.
  Value lookup(AllocaID allocaId) const {
    return static_cast<size_t>(allocaId) < allocaIdsToValuesMap.size()
               ? allocaIdsToValuesMap[allocaId]
               : Value();
  }

  /// Whether the analysis detected ill-formed alloca usage.
  bool isIllFormedIR() const { return illFormed; }

  /// Get the underlying data flow solver.
  const DataFlowSolver &getSolver() const { return solver; }

  /// Lookup the alloca alias state for a given value.
  const AllocaAlias *lookupState(Value v) const {
    auto *state = solver.lookupState<dataflow::Lattice<AllocaAlias>>(v);
    return state ? &state->getValue() : nullptr;
  }

  /// Get the alloca IDs for a given value.
  ArrayRef<AllocaID> getAllocaIds(Value v) const {
    auto *state = lookupState(v);
    return state ? state->getAllocaIds() : ArrayRef<AllocaID>();
  }

  /// Get the values corresponding to alloca IDs.
  ArrayRef<Value> getAllocas() const { return allocaIdsToValuesMap; }

protected:
  DenseMap<Value, AllocaID> valueToAllocaIdMap;
  SmallVector<Value> allocaIdsToValuesMap;
  const DataFlowSolver &solver;
  bool illFormed = false;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_ALLOCAALIASANALYSIS_H
