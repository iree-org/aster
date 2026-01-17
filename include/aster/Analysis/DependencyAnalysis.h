//===- DependencyAnalysis.h - Dependency analysis ----------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a forward dataflow analysis that tracks dependency tokens
// produced by operations implementing the DependentOpInterface. The analysis
// computes, for each program point, the set of outstanding (not-yet-waited-on)
// dependency tokens.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_DEPENDENCYANALYSIS_H
#define ASTER_ANALYSIS_DEPENDENCYANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include <cstddef>
#include <optional>

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// DependencyState
//===----------------------------------------------------------------------===//

/// This lattice represents the set of outstanding dependency tokens at a
/// program point. These are tokens that have been produced by memory operations
/// but have not yet been consumed by a wait operation.
struct DependencyState : dataflow::AbstractDenseLattice {
  using DependencySet = llvm::SmallDenseSet<Value>;
  DependencyState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), pendingTokens(DependencySet{}) {}

  /// Whether the state is the top state.
  bool isTop() const { return llvm::failed(pendingTokens); }

  /// Whether the state is empty.
  bool isEmpty() const {
    return llvm::succeeded(pendingTokens) && pendingTokens->empty();
  }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (isTop())
      return ChangeResult::NoChange;
    pendingTokens = failure();
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Join operation for the lattice.
  ChangeResult join(const DependencyState &lattice);
  ChangeResult join(const AbstractDenseLattice &lattice) final {
    return join(static_cast<const DependencyState &>(lattice));
  }

  /// Add tokens to the pending set.
  ChangeResult addTokens(ArrayRef<Value> tokens) {
    if (llvm::failed(pendingTokens))
      return ChangeResult::NoChange;
    size_t oldSize = pendingTokens->size();
    pendingTokens->insert_range(tokens);
    return pendingTokens->size() != oldSize ? ChangeResult::Change
                                            : ChangeResult::NoChange;
  }

  /// Remove tokens from the pending set (consumed by wait).
  ChangeResult removeTokens(ArrayRef<Value> tokens) {
    if (llvm::failed(pendingTokens))
      return ChangeResult::NoChange;
    bool changed = false;
    for (Value tok : tokens) {
      if (pendingTokens->erase(tok))
        changed = true;
    }
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  /// Get the pending tokens.
  const DependencySet *getPendingTokens() const {
    return succeeded(pendingTokens) ? &*pendingTokens : nullptr;
  }

private:
  FailureOr<DependencySet> pendingTokens;
};

//===----------------------------------------------------------------------===//
// DependencyAnalysis
//===----------------------------------------------------------------------===//

/// A forward dataflow analysis that tracks dependency tokens produced by
/// operations implementing DependentOpInterface. The analysis computes the
/// set of pending (not-yet-waited-on) dependency tokens at each program point.
class DependencyAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<DependencyState> {
public:
  using dataflow::DenseForwardDataFlowAnalysis<
      DependencyState>::DenseForwardDataFlowAnalysis;

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op, const DependencyState &before,
                               DependencyState *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const DependencyState &before,
                          DependencyState *after) override;

  /// Visit a call control flow transfer and update the lattice state.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const DependencyState &before,
                                    DependencyState *after) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const DependencyState &before,
                                            DependencyState *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(DependencyState *lattice) override;

private:
  /// Handle propagation when either of the states are top. Returns true if
  /// either state is top.
  bool handleTopPropagation(const DependencyState &before,
                            DependencyState *after);
};

} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::DependencyState)

#endif // ASTER_ANALYSIS_DEPENDENCYANALYSIS_H
