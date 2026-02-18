//===- DPSAnalysis.h - DPS analysis ------------------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_DPSANALYSIS_H
#define ASTER_ANALYSIS_DPSANALYSIS_H

#include "aster/IR/SSAMap.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
class DataFlowSolver;
} // namespace mlir

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// DPSAnalysis
//===----------------------------------------------------------------------===//

/// Analysis that tracks DPS provenance.
class DPSAnalysis {
public:
  struct Allocation {
    Allocation(Value value, SmallVector<int32_t, 1> &&ids)
        : value(value), ids(ids) {}
    Value value;
    SmallVector<int32_t, 1> ids;
  };

  struct AllocView {
    AllocView() = default;
    AllocView(Allocation *alloc, ArrayRef<int32_t> ids)
        : alloc(alloc), ids(ids) {}
    Allocation *alloc = nullptr;
    ArrayRef<int32_t> ids;
  };

  /// Provenance pair, the first element is the branching point operation, the
  /// second element is the value.
  using Provenance = std::pair<Operation *, Value>;
  using ProvenanceSet = llvm::SmallDenseSet<Provenance, 4>;

  /// Create a DPSAnalysis from an operation and dominance info.
  /// Returns failure if the analysis cannot be constructed.
  static FailureOr<DPSAnalysis> create(FunctionOpInterface funcOp);

  /// Get the value provenance map.
  llvm::DenseMap<Value, ProvenanceSet> &getProvenance() {
    return valueProvenance;
  }
  const llvm::DenseMap<Value, ProvenanceSet> &getProvenance() const {
    return valueProvenance;
  }

  /// Get the provenance set for a value. Returns nullptr if the value is not
  /// found.
  const ProvenanceSet *getProvenance(Value value) const {
    auto it = valueProvenance.find(value);
    if (it == valueProvenance.end())
      return nullptr;
    return &it->second;
  }

  /// Get or create a new allocation for a value, if the value is not a register
  /// type with value semantics, return nullptr.
  Allocation *getOrCreateAlloc(Value value);

  /// Get or create a new allocation for a value range. Returns nullptr on
  /// failure.
  Allocation *getOrCreateRange(Value value, ValueRange range);

  /// Get the allocation view for a value.
  AllocView getAllocView(Value value) const { return allocViews.lookup(value); }

  /// Maps a range of values to allocation ids. If value consists of a single
  /// value, it maps the value to the allocation. Otherwise it performs an
  /// element-wise mapping, and returns failure if the size of the range does
  /// not match the size of the allocation.
  LogicalResult mapAlloc(ValueRange values, Allocation *alloc);

  /// Print the analysis results (DPS alias equivalence classes and value
  /// provenance) for debugging and testing.
  void print(llvm::raw_ostream &os, const SSAMap &ssaMap) const;

  /// Get the allocation views map.
  DenseMap<Value, AllocView> &getAllocViews() { return allocViews; }
  const DenseMap<Value, AllocView> &getAllocViews() const { return allocViews; }

  /// Get the allocation that owns the given ID. Returns nullptr if the ID is
  /// unknown.
  const Allocation *getAllocForId(int32_t id) const {
    return idToAlloc.lookup_or(id, nullptr);
  }

private:
  /// Allocator for allocations.
  llvm::SpecificBumpPtrAllocator<Allocation> allocAllocator;

  /// Next available allocation ID.
  int64_t nextId = 0;

  /// Maps each value to its allocation view.
  DenseMap<Value, AllocView> allocViews;

  /// Inverse mapping from allocation ID to the allocation that owns it.
  llvm::DenseMap<int32_t, Allocation *> idToAlloc;

  /// Maps each control-flow variable to the set of values that constitute its
  /// provenance.
  llvm::DenseMap<Value, ProvenanceSet> valueProvenance;
};

//===----------------------------------------------------------------------===//
// DPSLiveness
//===----------------------------------------------------------------------===//

/// Liveness of DPS allocations at program points. Maps each operation to the
/// set of allocation IDs that are live at the after program point of that
/// instruction.
class DPSLiveness {
public:
  using LiveSet = llvm::SmallDenseSet<int32_t, 8>;
  /// Create a dps-aware liveness analysis. Returns failure if any liveness
  /// lattice cannot be procured for an operation in the function.
  static FailureOr<DPSLiveness> create(DPSAnalysis &dpsAnalysis,
                                       mlir::DataFlowSolver &solver,
                                       FunctionOpInterface funcOp);

  /// Return true iff any of the allocation IDs for values are live at the after
  /// program point of op. Return failure if op is not a valid program point
  /// (no liveness info).
  FailureOr<bool> areAnyLive(ValueRange values, Operation *op) const;

  /// Optional deterministic order of program points (walk order at creation).
  /// Empty if not populated.
  llvm::ArrayRef<Operation *> getOrderedProgramPoints() const {
    return orderedProgramPoints;
  }

  /// Print liveness: each program point with its sorted live allocation IDs.
  void print(llvm::raw_ostream &os, const SSAMap &ssaMap) const;

private:
  DPSLiveness(DPSAnalysis &dpsAnalysis) : dpsAnalysis(dpsAnalysis) {}
  DPSAnalysis &dpsAnalysis;
  llvm::DenseMap<Operation *, LiveSet> livenessInfo;
  llvm::SmallVector<Operation *, 0> orderedProgramPoints;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_DPSANALYSIS_H
