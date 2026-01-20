//===- LivenessAnalysis.h - Liveness analysis --------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_LIVENESSANALYSIS_H
#define ASTER_ANALYSIS_LIVENESSANALYSIS_H

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include <cstddef>

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// LivenessState
//===----------------------------------------------------------------------===//

/// This lattice represents liveness information using equivalence class IDs.
/// Each equivalence class ID maps 1-1 to an AllocaOp via DPSAliasAnalysis.
struct LivenessState : dataflow::AbstractDenseLattice {
  using LiveSet = llvm::SmallDenseSet<EqClassID>;
  LivenessState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), liveEqClasses(LiveSet{}) {}

  /// Whether the state is the top state.
  bool isTop() const { return llvm::failed(liveEqClasses); }

  /// Whether the state is empty.
  bool isEmpty() const {
    return llvm::succeeded(liveEqClasses) && liveEqClasses->empty();
  }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (isTop())
      return ChangeResult::NoChange;
    liveEqClasses = failure();
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Join operation for the lattice.
  ChangeResult meet(const LivenessState &lattice);
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const LivenessState &>(lattice));
  }

  /// Append equivalence class IDs to the live set.
  ChangeResult appendEqClassIds(ArrayRef<EqClassID> eqClassIds) {
    if (llvm::failed(liveEqClasses))
      return ChangeResult::NoChange;
    size_t oldSize = liveEqClasses->size();
    liveEqClasses->insert_range(eqClassIds);
    return liveEqClasses->size() != oldSize ? ChangeResult::Change
                                            : ChangeResult::NoChange;
  }

  /// Append equivalence class IDs to the live set.
  ChangeResult appendEqClassIds(const LiveSet &eqClassIds) {
    if (llvm::failed(liveEqClasses))
      return ChangeResult::NoChange;
    size_t oldSize = liveEqClasses->size();
    liveEqClasses->insert_range(eqClassIds);
    return liveEqClasses->size() != oldSize ? ChangeResult::Change
                                            : ChangeResult::NoChange;
  }

  /// Get the live equivalence class IDs.
  const LiveSet *getLiveEqClassIds() const {
    return succeeded(liveEqClasses) ? &*liveEqClasses : nullptr;
  }

private:
  FailureOr<LiveSet> liveEqClasses;
};

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// An analysis that, by going backwards along the dataflow graph, computes
/// liveness information. This analysis tracks live equivalence classes (which
/// map 1-1 to AllocaOps) rather than individual values, using DPSAliasAnalysis
/// to resolve value-to-equivalence-class mappings.
class LivenessAnalysis
    : public dataflow::DenseBackwardDataFlowAnalysis<LivenessState> {
public:
  LivenessAnalysis(DataFlowSolver &solver, SymbolTableCollection &symbolTable,
                   DPSAliasAnalysis *aliasAnalysis)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable),
        aliasAnalysis(aliasAnalysis) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op, const LivenessState &after,
                               LivenessState *before) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *successor,
                          const LivenessState &after,
                          LivenessState *before) override;

  /// Visit a call control flow transfer and update the lattice state.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const LivenessState &after,
                                    LivenessState *before) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            RegionBranchPoint regionFrom,
                                            RegionSuccessor regionTo,
                                            const LivenessState &after,
                                            LivenessState *before) override;

  /// Set the lattice to the entry state.
  void setToExitState(LivenessState *lattice) override;

private:
  /// Handle propagation when either of the states are top. Returns true if
  /// either state is top.
  bool handleTopPropagation(const LivenessState &after, LivenessState *before);

  /// Transfer function for liveness analysis. Takes dead and live equivalence
  /// class IDs and propagates them through the lattice.
  void transferFunction(const LivenessState &after, LivenessState *before,
                        SmallVector<EqClassID> &&deadEqClassIds,
                        ArrayRef<EqClassID> liveEqClassIds);

  /// Get the equivalence class IDs for a value. Returns empty if the value
  /// doesn't have a register type.
  ArrayRef<EqClassID> getEqClassIds(Value v) const;

  DPSAliasAnalysis *aliasAnalysis;
};
} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

#endif // ASTER_ANALYSIS_LIVENESSANALYSIS_H
