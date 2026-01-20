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
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// LivenessState
//===----------------------------------------------------------------------===//

/// This lattice represents liveness information using both values and
/// equivalence class IDs. We track individual values to determine when
/// an equivalence class can become dead: only when ALL values aliasing
/// that class are dead.
struct LivenessState : dataflow::AbstractDenseLattice {
  using ValueSet = llvm::SmallDenseSet<Value>;
  using EqClassSet = llvm::SmallDenseSet<EqClassID>;
  LivenessState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), isTopState(false), liveValues(),
        liveEqClasses() {}

  /// Whether the state is the top state.
  bool isTop() const { return isTopState; }

  /// Whether the state is empty.
  bool isEmpty() const { return !isTopState && liveValues.empty(); }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (isTopState)
      return ChangeResult::NoChange;
    isTopState = true;
    liveValues.clear();
    liveEqClasses.clear();
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Join operation for the lattice.
  ChangeResult meet(const LivenessState &lattice);
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const LivenessState &>(lattice));
  }

  /// Get the live values. Requires that the state is not top.
  const ValueSet &getLiveValues() const {
    assert(!isTopState && "Cannot get live values from top state");
    return liveValues;
  }

  /// Get the live equivalence class IDs. Requires that the state is not top.
  const EqClassSet &getLiveEqClassIds() const {
    assert(!isTopState && "Cannot get live eq class IDs from top state");
    return liveEqClasses;
  }

  /// Update the state with new live values and their eq classes.
  /// Returns Change if the state was modified.
  /// This method does nothing if the state is top.
  ChangeResult updateLiveness(const ValueSet &newValues,
                              const EqClassSet &newEqClasses) {
    if (isTopState)
      return ChangeResult::NoChange;
    size_t oldValSize = liveValues.size();
    size_t oldEqSize = liveEqClasses.size();
    liveValues.insert_range(newValues);
    liveEqClasses.insert_range(newEqClasses);
    return (liveValues.size() != oldValSize ||
            liveEqClasses.size() != oldEqSize)
               ? ChangeResult::Change
               : ChangeResult::NoChange;
  }

private:
  bool isTopState;
  ValueSet liveValues;
  EqClassSet liveEqClasses;
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

  /// Transfer function for liveness analysis.
  /// - deadValues: Results being defined (dead going backwards)
  /// - liveValues: Operands always live with their eq classes
  /// - aliasingOperands: Operands live but eq classes conditional on callback
  /// - isEqClassLive: Returns true if aliasing operand's eq class should be
  ///   live (only called for aliasingOperands)
  void transferFunction(const LivenessState &after, LivenessState *before,
                        ArrayRef<Value> deadValues, ArrayRef<Value> liveValues,
                        ArrayRef<Value> aliasingOperands = {},
                        function_ref<bool(Value)> isEqClassLive = nullptr);

  DPSAliasAnalysis *aliasAnalysis;
};
} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

#endif // ASTER_ANALYSIS_LIVENESSANALYSIS_H
