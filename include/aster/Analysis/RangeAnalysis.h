//===- RangeAnalysis.h - Range analysis -------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the register range analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_RANGEANALYSIS_H
#define ASTER_ANALYSIS_RANGEANALYSIS_H

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Support/Graph.h"
#include "mlir/Support/LLVM.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace mlir::aster {

namespace amdgcn {
class AllocaOp;
class MakeRegisterRangeOp;
} // namespace amdgcn
/// Represents an irreducible range of registers.
struct Range {
  Range(amdgcn::MakeRegisterRangeOp rangeOp, ArrayRef<EqClassID> eqClassIds)
      : rangeOp(rangeOp), eqClassIds(eqClassIds) {}

  /// Get the equivalence class IDs that make up this range.
  ArrayRef<EqClassID> getEqClassIds() const { return eqClassIds; }

  /// Get the register type of this range.
  RegisterTypeInterface getRegisterType() const { return rangeOp.getType(); }

  /// Get the underlying operation of this range.
  amdgcn::MakeRegisterRangeOp getOp() const { return rangeOp; }

  /// Get the start equivalence class of this range.
  EqClassID startEqClassId() const { return eqClassIds.front(); }

private:
  mutable amdgcn::MakeRegisterRangeOp rangeOp;
  ArrayRef<EqClassID> eqClassIds;
};

/// Represents an allocation of a range of registers.
struct RangeAllocation {
  RangeAllocation(EqClassID start) { allocatedEqClassIds.insert(start); }

  /// Add an equivalence class to the allocation.
  void pushEqClassId(EqClassID eqClassId) {
    allocatedEqClassIds.insert(eqClassId);
  }

  /// Get the allocated equivalence classes.
  ArrayRef<EqClassID> getAllocatedEqClassIds() const {
    return allocatedEqClassIds.getArrayRef();
  }

  /// Get the size of the allocation.
  size_t size() const { return allocatedEqClassIds.size(); }

  /// Get the alignment of the allocation.
  int32_t getAlignment() const { return alignment; }

  /// Set alignment constraints.
  LogicalResult setAlignment(Location loc, EqClassID eqClassId,
                             int32_t alignCtr,
                             amdgcn::MakeRegisterRangeOp owner);

  /// Get the start equivalence class of this range.
  EqClassID startEqClassId() const { return allocatedEqClassIds.front(); }

private:
  SetVector<EqClassID> allocatedEqClassIds;
  int32_t alignment = 1;
};

//===----------------------------------------------------------------------===//
// RangeAnalysis
//===----------------------------------------------------------------------===//
/// This class represents the register range analysis.
struct RangeAnalysis {
  /// Create a RangeAnalysis instance from a DPSAliasAnalysis.
  static RangeAnalysis create(Operation *topOp,
                              const DPSAliasAnalysis *analysis);

  /// Returns true if the range constraints are satisfiable.
  bool isSatisfiable() const { return succeeded(allocationMap); }

  /// Get the underlying alias analysis.
  const DPSAliasAnalysis *getAnalysis() const { return analysis; }

  /// Get the underlying graph.
  const Graph &getGraph() const { return graph; }

  /// Get the ranges.
  ArrayRef<Range> getRanges() const { return ranges; }

  /// Get the allocation constraint or nullptr if the equivalence class is not
  /// tied to any ranges.
  const RangeAllocation *lookupAllocation(EqClassID eqClassId) const {
    assert(isSatisfiable() && "Range constraints are not satisfiable");
    int32_t allocId = allocationMap->lookup_or(eqClassId, -1);
    if (allocId == -1)
      return nullptr;
    return &allocations[allocId];
  }

  /// Get the allocations.
  ArrayRef<RangeAllocation> getAllocations() const {
    assert(isSatisfiable() && "Range constraints are not satisfiable");
    return allocations;
  }

private:
  RangeAnalysis(const DPSAliasAnalysis *analysis)
      : analysis(analysis), graph(true) {}
  const DPSAliasAnalysis *analysis;
  SmallVector<Range> ranges;
  SmallVector<RangeAllocation> allocations;
  Graph graph;
  FailureOr<DenseMap<EqClassID, int32_t>> allocationMap;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_RANGEANALYSIS_H
