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
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
class DataFlowSolver;
} // namespace mlir

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// DPSAnalysis
//===----------------------------------------------------------------------===//

/// Analysis that tracks DPS provenance.
/// Creates fresh Alocation objects on AllocaOpInterface, MakeRegisterRangeOp
/// and control-flow edges. All operations add AllocView on top of existing
/// Allocation objects. Users of DPSAnalysis only manipulate AllocView.
/// DPS analysis also records control-flow provenance:
///   which (branch-point, value) pairs feed each bbarg.
class DPSAnalysis {
public:
  /// Created on every:
  ///   - AllocaOpInterface
  ///   - visitControlFlowEdge on Successor inputs
  ///   - getOrCreateRange on make_register_range
  /// with 1 value and #ids == register range size.
  /// Represents the underlying storage for each root allocation.
  /// Users of DPSAnalysis only manipulate AllocView.
  struct Allocation {
    Allocation(Value value, SmallVector<int32_t, 1> &&ids)
        : value(value), ids(ids) {}
    Value value;
    SmallVector<int32_t, 1> ids;
  };

  /// Canonical AllocView created along every Allocation.
  /// Propagated from existing Allocation on either:
  /// - InstOpInterface TiedInstOutsRange
  /// - split_register_range (slice)
  /// Users only need to manipulate AllocView.
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
  /// Set the allocation view for a value.
  void setAllocView(Value value, AllocView &&allocView) {
    allocViews[value] = allocView;
  }

  /// Print the analysis results (DPS alias equivalence classes and value
  /// provenance) for debugging and testing.
  void print(llvm::raw_ostream &os, const SSAMap &ssaMap) const;

  /// Get the allocation views map.
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
  int32_t nextId = 0;

  /// Maps each value to its allocation view.
  DenseMap<Value, AllocView> allocViews;

  /// Inverse mapping from allocation ID to the allocation that owns it.
  llvm::DenseMap<int32_t, Allocation *> idToAlloc;

  /// Maps each control-flow variable to the set of values that constitute its
  /// provenance.
  llvm::DenseMap<Value, ProvenanceSet> valueProvenance;
};

//===----------------------------------------------------------------------===//
// DPSClobberingAnalysis
//===----------------------------------------------------------------------===//

/// DPS clobbering analysis. This analysis indicates whether an instruction
/// result with register value semantics requires de-clobbering.
class DPSClobberingAnalysis {
public:
  /// Create a clobbering analysis from a DPS analysis and a dataflow solver
  /// with liveness information. Returns failure if the analysis cannot be
  /// constructed.
  static FailureOr<DPSClobberingAnalysis> create(DPSAnalysis &dpsAnalysis,
                                                 mlir::DataFlowSolver &solver,
                                                 FunctionOpInterface funcOp);

  /// Returns an array of booleans indicating whether a tied out/result requires
  /// de-clobbering. The array only contains entries for the instruction
  /// results.
  ArrayRef<bool> getClobberingInfo(InstOpInterface op) const {
    auto it = clobberingInfo.find(op);
    if (it == clobberingInfo.end())
      return ArrayRef<bool>();
    return it->second;
  }

private:
  DPSClobberingAnalysis(DPSAnalysis &dpsAnalysis) : dpsAnalysis(dpsAnalysis) {}
  DPSAnalysis &dpsAnalysis;
  llvm::DenseMap<Operation *, SmallVector<bool, 4>> clobberingInfo;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_DPSANALYSIS_H
