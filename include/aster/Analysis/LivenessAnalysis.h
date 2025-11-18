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

/// This lattice represents liveness information.
struct LivenessState : dataflow::AbstractDenseLattice {
  using LiveSet = llvm::SmallDenseSet<Value>;
  LivenessState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), liveValues(LiveSet{}) {}

  /// Whether the state is the top state.
  bool isTop() const { return llvm::failed(liveValues); }

  /// Whether the state is empty.
  bool isEmpty() const {
    return llvm::succeeded(liveValues) && liveValues->empty();
  }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (isTop())
      return ChangeResult::NoChange;
    liveValues = failure();
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Join operation for the lattice.
  ChangeResult meet(const LivenessState &lattice);
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const LivenessState &>(lattice));
  }

  /// Append values to the live set.
  ChangeResult appendValues(ArrayRef<Value> values) {
    if (llvm::failed(liveValues))
      return ChangeResult::NoChange;
    size_t oldSize = liveValues->size();
    liveValues->insert_range(values);
    return liveValues->size() != oldSize ? ChangeResult::Change
                                         : ChangeResult::NoChange;
  }

  /// Append values to the live set.
  ChangeResult appendValues(const LiveSet &values) {
    if (llvm::failed(liveValues))
      return ChangeResult::NoChange;
    size_t oldSize = liveValues->size();
    liveValues->insert_range(values);
    return liveValues->size() != oldSize ? ChangeResult::Change
                                         : ChangeResult::NoChange;
  }

  /// Get the live values.
  const LiveSet *getLiveValues() const {
    return succeeded(liveValues) ? &*liveValues : nullptr;
  }

private:
  FailureOr<LiveSet> liveValues;
};

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// An analysis that, by going backwards along the dataflow graph, computes
/// liveness information.
class LivenessAnalysis
    : public dataflow::DenseBackwardDataFlowAnalysis<LivenessState> {
public:
  using dataflow::DenseBackwardDataFlowAnalysis<
      LivenessState>::DenseBackwardDataFlowAnalysis;

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
  void transferFunction(const LivenessState &after, LivenessState *before,
                        SmallVector<Value> &&deadValues, ValueRange inValues);
};
} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::LivenessState)

#endif // ASTER_ANALYSIS_LIVENESSANALYSIS_H
