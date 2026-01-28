//===- BufferAnalysis.h - Buffer analysis ------------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis computes, for each program point, the state of reachable
// buffers.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_BUFFERANALYSIS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_BUFFERANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"

namespace mlir::aster::amdgcn {
//===----------------------------------------------------------------------===//
// BufferState
//===----------------------------------------------------------------------===//

/// This lattice represents the set of buffers at a program point.
/// A buffer can be in one of the following states:
/// - None: The buffer is not present in the lattice.
/// - Live: The buffer is live (active lifetime).
/// - Dead: The buffer is dead (lifetime has ended).
/// - Top: The buffer state is unknown (conflicting information).
struct BufferState : dataflow::AbstractDenseLattice {
  /// Represents the state of a buffer.
  enum class State : uint8_t { None, Live, Dead, Top };
  using BufferMap = llvm::SmallDenseMap<Value, State, 4>;

  BufferState(LatticeAnchor anchor) : AbstractDenseLattice(anchor) {}

  /// Whether the state is empty (no live buffers).
  bool isEmpty() const { return buffers.empty(); }

  /// Join operation for the lattice. If a dominance function is provided,
  /// buffers that do not dominate are skipped.
  ChangeResult join(const BufferState &lattice,
                    llvm::function_ref<bool(Value)> dominates = {});
  ChangeResult join(const AbstractDenseLattice &lattice) final {
    return join(static_cast<const BufferState &>(lattice));
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Add a buffer to the set. An assertion is raised if the buffer already
  /// exists.
  ChangeResult addBuffer(Value buffer) {
    auto [it, inserted] = buffers.insert({buffer, State::Live});
    if (it->second != State::Live) {
      it->second = State::Top;
      return ChangeResult::Change;
    }
    return inserted ? ChangeResult::Change : ChangeResult::NoChange;
  }

  /// Kill a buffer in the set.
  ChangeResult killBuffer(Value buffer) {
    State &state = buffers[buffer];
    State cur = state;

    // If the state is Live, mark it as Dead, and if it is not dead, mark it as
    // Top.
    if (cur == State::Live)
      state = State::Dead;
    else if (cur != State::Dead)
      state = State::Top;
    return cur != state ? ChangeResult::Change : ChangeResult::NoChange;
  }

  /// Mark a buffer as top in the set.
  ChangeResult markAsTop(Value buffer) {
    State &state = buffers[buffer];
    return std::exchange(state, State::Top) != state ? ChangeResult::Change
                                                     : ChangeResult::NoChange;
  }

  /// Get the state of a buffer.
  State getBufferState(Value buffer) const {
    return buffers.lookup_or(buffer, State::None);
  }

  /// Get the map of buffers to their states.
  const BufferMap &getBuffers() const { return buffers; }

  /// Mark all buffers as top in the set.
  ChangeResult markAllAsTop() {
    ChangeResult result = ChangeResult::NoChange;
    for (auto it : buffers) {
      State &state = it.second;
      State cur = state;
      state = State::Top;
      if (cur != state)
        result = ChangeResult::Change;
    }
    return result;
  }

  /// Update the state of a buffer.
  ChangeResult updateBuffer(Value buffer, State state);

private:
  BufferMap buffers;
};

//===----------------------------------------------------------------------===//
// BufferAnalysis
//===----------------------------------------------------------------------===//

/// A forward dataflow analysis that tracks the buffer state at each program
/// point.
class BufferAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<BufferState> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<BufferState>;
  using Base::Base;
  BufferAnalysis(DataFlowSolver &solver, DominanceInfo &domInfo)
      : Base(solver), domInfo(domInfo) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op, const BufferState &before,
                               BufferState *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const BufferState &before,
                          BufferState *after) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const BufferState &before,
                                            BufferState *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(BufferState *lattice) override;

private:
  /// Reference to the dominance info.
  DominanceInfo &domInfo;
};
} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::BufferState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_BUFFERANALYSIS_H
