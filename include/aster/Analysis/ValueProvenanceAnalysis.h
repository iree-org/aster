//===- ValueProvenanceAnalysis.h - Value provenance analysis ----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a sparse forward dataflow analysis that tracks the
// provenance of SSA values - specifically, which alloca operation a value
// originates from.
//
// Precondition: The IR must be in a form where all register-typed values
// flow through AllocaOp -> InstOpInterface chains. Use verifyIRForm() to
// check this before running the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_VALUEPROVENANCEANALYSIS_H
#define ASTER_ANALYSIS_VALUEPROVENANCEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// ValueProvenance
//===----------------------------------------------------------------------===//

/// Lattice tracking originating alloca for a value.
/// States: bottom (uninitialized), known alloca(s), top (unknown).
class ValueProvenance {
public:
  /// Construct an uninitialized provenance (bottom).
  ValueProvenance() : provenanceAllocas(llvm::SetVector<Value>{}) {}

  /// Construct a provenance with a known alloca.
  explicit ValueProvenance(Value allocaOp)
      : provenanceAllocas(llvm::SetVector<Value>{}) {
    provenanceAllocas->insert(allocaOp);
  }

  /// Get the uninitialized state (bottom).
  static ValueProvenance getUninitialized() { return ValueProvenance(); }

  /// Get the top state (unknown).
  static ValueProvenance getTop() { return ValueProvenance(std::nullopt); }

  /// Whether this is the uninitialized state.
  bool isUninitialized() const {
    return succeeded(provenanceAllocas) && provenanceAllocas->empty();
  }

  /// Whether this is the top state.
  bool isTop() const { return failed(provenanceAllocas); }

  /// Get the first alloca in the set. Returns null if uninitialized or top.
  Value getFirstAllocaInSet() const {
    return succeeded(provenanceAllocas) && !provenanceAllocas->empty()
               ? provenanceAllocas->front()
               : Value();
  }

  /// For phi-coalesced values that merged at block arguments.
  ArrayRef<Value> getAllocas() const {
    return succeeded(provenanceAllocas) ? provenanceAllocas->getArrayRef()
                                        : ArrayRef<Value>();
  }

  /// Join operation for lattice (used at control flow merge points).
  static ValueProvenance join(const ValueProvenance &lhs,
                              const ValueProvenance &rhs);

  /// Equality comparison.
  bool operator==(const ValueProvenance &rhs) const {
    return succeeded(provenanceAllocas) == succeeded(rhs.provenanceAllocas) &&
           (failed(provenanceAllocas) ||
            *provenanceAllocas == *rhs.provenanceAllocas);
  }

  /// Print the provenance value.
  void print(raw_ostream &os) const;

private:
  /// Construct a top state.
  explicit ValueProvenance(std::nullopt_t) : provenanceAllocas(failure()) {}

  /// State encoding: bottom (empty), known allocas (non-empty), top (failure).
  FailureOr<llvm::SetVector<Value>> provenanceAllocas;
};

inline raw_ostream &operator<<(raw_ostream &os, const ValueProvenance &vp) {
  vp.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// ValueProvenanceAnalysis
//===----------------------------------------------------------------------===//

/// Forward sparse dataflow analysis tracking value provenance.
///
/// Collects allocas that merge at block arguments (phi-coalescing) and computes
/// phi-equivalences via union-find.

/// DPSAliasAnalysis consumes these phi-equivalences to assign consistent IDs to
/// phi-coalesced values.
/// Regalloc uses these phi-equivalences to introduce additional allocations and
/// copies where the interference graph and phi-equivalence would conflict.
///
/// Correctness: All allocas merging at a block argument must be marked
/// equivalent by finalizeEquivalences(), or DPSAliasAnalysis will report
/// ill-formed IR.
class ValueProvenanceAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                                    dataflow::Lattice<ValueProvenance>> {
public:
  /// Create, run, and finalize provenance analysis.
  static ValueProvenanceAnalysis *create(DataFlowSolver &solver,
                                         Operation *topOp);

  /// Get the canonical alloca for a value. Returns failure if provenance is
  /// unknown (top or uninitialized).
  FailureOr<Value> getCanonicalPhiEquivalentAlloca(Value v) const;

  /// Check if two allocas are phi-equivalent (must be merged at block arg).
  bool arePhiEquivalent(Value a, Value b) const;

  /// Get all allocas phi-equivalent to the given one.
  /// Returns empty if the alloca has no known phi-equivalences.
  SmallVector<Value> getPhiEquivalentAllocas(Value alloca) const;

  /// Returns failure if unsupported ops (from_reg/to_reg) are found.
  static LogicalResult verifyIRForm(Operation *topOp);

private:
  explicit ValueProvenanceAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver), solver(solver) {}

  // Reference to the dataflow solver.
  const DataFlowSolver &solver;

  /// Index-based union-find for deterministic ordering (indices assigned
  /// during IR traversal, so iteration order is stable across runs).
  DenseMap<Value, int64_t> allocaToIndex;
  SmallVector<Value> indexToAlloca;
  llvm::EquivalenceClasses<int64_t> phiEquivalences;

  // Used for solver.load<ValueProvenanceAnalysis>() during initialization.
  friend class mlir::DataFlowSolver;

  // Dataflow analysis implementation methods.
  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::Lattice<ValueProvenance> *> operands,
      ArrayRef<dataflow::Lattice<ValueProvenance> *> results) override;

  /// Set the entry state for a lattice (used at function entry, etc.).
  void setToEntryState(dataflow::Lattice<ValueProvenance> *lattice) override;
};

} // namespace mlir::aster

#endif // ASTER_ANALYSIS_VALUEPROVENANCEANALYSIS_H
