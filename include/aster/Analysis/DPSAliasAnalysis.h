//===- DPSAliasAnalysis.h - DPS alias analysis -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements variable analysis using sparse data-flow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_DPSALIASANALYSIS_H
#define ASTER_ANALYSIS_DPSALIASANALYSIS_H

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
// Variable
//===----------------------------------------------------------------------===//

using VariableID = int32_t;
/// This lattice value represents variable information.
class Variable {
public:
  using VIdList = llvm::SmallVector<VariableID, 4>;
  /// Construct a variable value.
  Variable(VIdList variableIds = {}) : variableIds(std::move(variableIds)) {}

  /// Compare variable values.
  bool operator==(const Variable &rhs) const {
    return succeeded(rhs.variableIds) == succeeded(variableIds) &&
           (failed(rhs.variableIds) || *rhs.variableIds == *variableIds);
  }

  /// Print the variable value.
  void print(raw_ostream &os) const;

  /// The state where the variable value is uninitialized.
  static Variable getUninitialized() { return Variable{}; }

  /// The state where the variable value is overdefined.
  static Variable getTop() { return Variable(std::nullopt); }

  /// Whether the state is uninitialized.
  bool isUninitialized() const {
    return llvm::succeeded(variableIds) && variableIds->empty();
  }

  /// Whether the state is top (overdefined).
  bool isTop() const { return llvm::failed(variableIds); }

  /// Join operation for lattice.
  static Variable join(const Variable &lhs, const Variable &rhs) {
    if (lhs.isTop() || rhs.isTop())
      return getTop();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (*lhs.variableIds == *rhs.variableIds)
      return lhs;
    /// If the variable IDs differ, we conservatively return top.
    return Variable(std::nullopt);
  }

  /// Get the variable IDs. Returns an empty list if uninitialized or top.
  ArrayRef<VariableID> getVariableIds() const {
    return succeeded(variableIds) ? *variableIds : ArrayRef<VariableID>();
  }

private:
  /// Construct a variable value from a LogicalResult. This only works for
  /// failure states.
  explicit Variable(std::nullopt_t) : variableIds(failure()) {}
  llvm::FailureOr<VIdList> variableIds;
};

//===----------------------------------------------------------------------===//
// DPSAliasAnalysis
//===----------------------------------------------------------------------===//

/// This analysis implements variable analysis using sparse forward data-flow
/// analysis.
class DPSAliasAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                             dataflow::Lattice<Variable>> {
public:
  DPSAliasAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver), solver(solver) {}
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<Variable> *> operands,
                 ArrayRef<dataflow::Lattice<Variable> *> results) override;

  void setToEntryState(dataflow::Lattice<Variable> *lattice) override;

  /// Lookup the variable ID assigned to the given value. Returns -1 if not
  /// found.
  VariableID lookup(Value val) const {
    return valueToVarIdMap.lookup_or(val, -1);
  }

  /// Lookup the value assigned to the given variable ID. Returns null Value if
  /// not found.
  Value lookup(VariableID varId) const {
    return static_cast<size_t>(varId) < idsToValuesMap.size()
               ? idsToValuesMap[varId]
               : Value();
  }

  /// Whether the analysis detected ill-formed variable usage.
  bool isIllFormedIR() const { return illFormed; }

  /// Get the underlying data flow solver.
  const DataFlowSolver &getSolver() const { return solver; }

  /// Lookup the variable state for a given value.
  const Variable *lookupState(Value v) const {
    auto *state = solver.lookupState<dataflow::Lattice<Variable>>(v);
    return state ? &state->getValue() : nullptr;
  }

  /// Get the variable IDs for a given value.
  ArrayRef<VariableID> getVariableIds(Value v) const {
    auto *state = lookupState(v);
    return state ? state->getVariableIds() : ArrayRef<VariableID>();
  }

  /// Get the values corresponding to variable IDs.
  ArrayRef<Value> getVariables() const { return idsToValuesMap; }

protected:
  DenseMap<Value, VariableID> valueToVarIdMap;
  SmallVector<Value> idsToValuesMap;
  const DataFlowSolver &solver;
  bool illFormed = false;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_DPSALIASANALYSIS_H
