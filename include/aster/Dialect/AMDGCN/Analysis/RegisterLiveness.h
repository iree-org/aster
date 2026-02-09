//===- RegisterLiveness.h - Register liveness analysis -----------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERLIVENESS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERLIVENESS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::aster {
class SSAMap;
namespace amdgcn {

//===----------------------------------------------------------------------===//
// RegisterLivenessState
//===----------------------------------------------------------------------===//

/// This lattice represents register liveness information.
struct RegisterLivenessState : dataflow::AbstractDenseLattice {
  using ValueSet = llvm::SmallPtrSet<Value, 4>;
  RegisterLivenessState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), liveValues(ValueSet()) {}

  /// Whether the state is the top state.
  bool isTop() const { return failed(liveValues); }

  /// Whether the state is empty.
  bool isEmpty() const { return !isTop() && liveValues->empty(); }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (isTop())
      return ChangeResult::NoChange;
    liveValues = failure();
    return ChangeResult::Change;
  }

  /// Set the lattice to empty.
  ChangeResult setToEmpty() {
    if (isEmpty())
      return ChangeResult::NoChange;
    liveValues = ValueSet();
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;
  void print(raw_ostream &os, const SSAMap &ssaMap) const;

  /// Meet operation for the lattice.
  ChangeResult meet(const RegisterLivenessState &lattice);
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const RegisterLivenessState &>(lattice));
  }

  /// Get the live values. Returns nullptr if the state is top.
  ValueSet *getLiveValues() {
    return succeeded(liveValues) ? &*liveValues : nullptr;
  }

  const ValueSet *getLiveValues() const {
    return succeeded(liveValues) ? &*liveValues : nullptr;
  }

private:
  FailureOr<llvm::SmallPtrSet<Value, 4>> liveValues;
};

//===----------------------------------------------------------------------===//
// RegisterLiveness
//===----------------------------------------------------------------------===//

/// An analysis that, by going backwards along the dataflow graph, computes
/// register liveness information.
class RegisterLiveness
    : public dataflow::DenseBackwardDataFlowAnalysis<RegisterLivenessState> {
public:
  RegisterLiveness(DataFlowSolver &solver, SymbolTableCollection &symbolTable)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op,
                               const RegisterLivenessState &after,
                               RegisterLivenessState *before) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *successor,
                          const RegisterLivenessState &after,
                          RegisterLivenessState *before) override;

  /// Visit a call control flow transfer and update the lattice state.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const RegisterLivenessState &after,
                                    RegisterLivenessState *before) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
      RegionSuccessor regionTo, const RegisterLivenessState &after,
      RegisterLivenessState *before) override;

  /// Set the lattice to the exit state.
  void setToExitState(RegisterLivenessState *lattice) override;

  /// Return true if the liveness analysis is incomplete. This is raised if
  /// value semantics are detected.
  bool isIncompleteLiveness() const { return incompleteLiveness; }

private:
  /// Handle propagation when either of the states are top. Returns true if
  /// either state is top.
  bool handleTopPropagation(const RegisterLivenessState &after,
                            RegisterLivenessState *before);

  /// Transfer function for liveness analysis.
  void transferFunction(const RegisterLivenessState &after,
                        RegisterLivenessState *before, ValueRange deadValues,
                        ValueRange inValues);
  bool incompleteLiveness = false;
};
} // namespace amdgcn
} // namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::RegisterLivenessState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERLIVENESS_H
