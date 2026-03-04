//===- HazardAnalysis.h - Hazard dataflow analysis --------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forward dataflow analysis that tracks active hazards at each program point,
// and the instruction counts that were flushed to resolve them.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_HAZARDANALYSIS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_HAZARDANALYSIS_H

#include "aster/Dialect/AMDGCN/IR/HazardManager.h"
#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"

namespace mlir::aster::amdgcn {
//===----------------------------------------------------------------------===//
// HazardState
//===----------------------------------------------------------------------===//

/// Lattice element for hazard analysis: vector of active hazards (sorted) and
/// a small dense set of consumed hazards.
struct HazardState : dataflow::AbstractDenseLattice {
  HazardState(LatticeAnchor anchor) : AbstractDenseLattice(anchor) {}

  /// Whether the state is empty.
  bool isEmpty() const {
    return activeHazards.empty() && flushedInstCounts.isInactive();
  }

  /// Meet (join) operation: merge-sorts active hazards and uses Hazard::join
  /// when equivalent hazards are found.
  ChangeResult join(const HazardState &lattice);
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    return join(static_cast<const HazardState &>(lattice));
  }

  /// Add hazards produced by an operation.
  ChangeResult addHazards(ArrayRef<Hazard> hazards);

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Print the lattice element in a deterministic order.
  void printDeterministic(raw_ostream &os, DominanceInfo &domInfo) const;

  /// Active (reaching) hazards at this program point, kept sorted for merge.
  SmallVector<Hazard> activeHazards;
  /// The instruction counts that were flushed at this program point.
  InstCounts flushedInstCounts;
};

//===----------------------------------------------------------------------===//
// HazardAnalysis
//===----------------------------------------------------------------------===//

/// Forward dataflow analysis that tracks active hazards at each program point.
/// Receives a HazardManager reference to compute hazards for instructions.
class HazardAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<HazardState> {
public:
  using dataflow::DenseForwardDataFlowAnalysis<
      HazardState>::DenseForwardDataFlowAnalysis;

  HazardAnalysis(DataFlowSolver &solver, HazardManager &hazardManager)
      : dataflow::DenseForwardDataFlowAnalysis<HazardState>(solver),
        hazardManager(hazardManager) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op, const HazardState &before,
                               HazardState *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const HazardState &before,
                          HazardState *after) override;

  /// Visit a call control flow transfer.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const HazardState &before,
                                    HazardState *after) override;

  /// Visit a region branch control flow transfer.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const HazardState &before,
                                            HazardState *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(HazardState *lattice) override;

private:
  HazardManager &hazardManager;
};

} // namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::HazardState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_HAZARDANALYSIS_H
