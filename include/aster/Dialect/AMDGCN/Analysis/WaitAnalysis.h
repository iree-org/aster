//===- WaitAnalysis.h - Wait dependency analysis ----------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis computes, for each program point, the set of outstanding
// (not-yet-waited-on) dependency tokens. It is implemented as a dense forward
// dataflow analysis.
//
// The analysis tracks dependency tokens produced by operations implementing
// DependentOpInterface (load/store operations) and consumed via WaitOps. Key
// capabilities include:
//
// - Token production tracking from load/store operations
// - Token consumption via WaitOps with position-based counting
// - Position updates as new tokens are pushed (stack-like ordering for FIFO
//   hardware counters)
// - Control flow merging with dominance-aware token filtering
// - Region-based control flow transfers
//
// The analysis is built on several core components:
//
// - TokenState: Represents a dependency token with its ID, position, and
//   token kind.
//
// - WaitCnt: Helper struct for managing hardware wait count values (vm_cnt,
//   lgkm_cnt). Provides utilities for computing counts from WaitOps and
//   updating counts based on token positions.
//
// - WaitState: Lattice element tracking reaching tokens at each program point,
//   plus optional WaitOpInfo that captures waited tokens, implied tokens, and
//   computed wait counts.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir::aster::amdgcn {
enum class ISAVersion : uint32_t;
class WaitOp;
class WaitCntOpInterface;
class WaitAnalysis;
struct WaitState;

/// The hardware wait-counter model determines how individual counter kinds
/// map to physical counters.
/// Before GFX12_50, load+store share (alias on) vmcnt and ds+km share lgkmcnt.
/// In GFX12_50 each kind is an independent counter.
enum class WaitCounterModel : uint8_t {
  CDNA3,
  CDNA4,
  GFX12_50,
};

/// Map an ISAVersion to the corresponding WaitCounterModel.
/// Only CDNA3, CDNA4, and GFX12_50 are supported.
WaitCounterModel getWaitCounterModel(ISAVersion isa);

/// Resolve the WaitCounterModel for an arbitrary op by looking for an enclosing
/// amdgcn.module (or the op itself if it is one). Falls back to CDNA3 when no
/// module is found. Use this everywhere a model is needed from an op context.
WaitCounterModel getWaitCounterModelForOp(Operation *op);

//===----------------------------------------------------------------------===//
// TokenState
//===----------------------------------------------------------------------===//

/// A token state represents a dependency token along with its kind and
/// position.
struct TokenState {
  using ID = int32_t;
  using Position = uint16_t;
  static constexpr ID kUnknownID = std::numeric_limits<ID>::max();
  static constexpr Position kMinPosition = std::numeric_limits<Position>::min();
  static constexpr Position kMaxPosition = std::numeric_limits<Position>::max();

  TokenState() = default;
  TokenState(Value token, ID id, WaitCounterKind kind, Position position = 0)
      : token(token), id(id), position(position), kind(kind) {}

  /// Create an unknown token state for a given counter kind.
  static TokenState unknown(WaitCounterKind kind, Position pos) {
    return TokenState(nullptr, kUnknownID, kind, pos);
  }

  /// Equality operator for TokenState.
  bool operator==(Value value) const { return token == value; }
  bool operator==(const TokenState &other) const {
    return token == other.token && kind == other.kind && id == other.id &&
           position == other.position;
  }

  /// Less-than operator for TokenState, used for sorting.
  bool operator<(const TokenState &other) const {
    if (kind != other.kind)
      return kind < other.kind;
    return id < other.id;
  }

  /// Advance the state.
  TokenState &operator++();

  /// Merge another token state into this one.
  bool merge(const TokenState &other);
  /// Return the token value.
  Value getToken() const { return token; }
  /// Returns the unique ID of the token.
  ID getID() const { return id; }
  /// Returns the position of the token.
  int32_t getPosition() const { return position; }
  /// Returns the hardware wait-counter kind of the token.
  WaitCounterKind getKind() const { return kind; }
  /// Print the state with model-specific kind names.
  void print(raw_ostream &os, WaitCounterModel model) const;
  /// Print with GFX12_50 names (the superset, for generic dump).
  void print(raw_ostream &os) const { print(os, WaitCounterModel::GFX12_50); }

private:
  Value token = nullptr;
  ID id = -1;
  Position position = 0;
  WaitCounterKind kind = WaitCounterKind::Load;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TokenState &state) {
  state.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

/// Object representing the analysis's per-counter-group wait counts.
struct WaitCnt {
  using Position = TokenState::Position;
  static constexpr Position kMaxPosition = TokenState::kMaxPosition;
  /// One position kind per WaitCounterKind.
  /// Use gfx1250 load/store/ds/km/tensor and project CDNA vm/lgkm onto it.
  static constexpr size_t kNumKinds =
      static_cast<size_t>(WaitCounterKind::Tensor) + 1;

  WaitCnt() { counts.fill(kMaxPosition); }

  /// Get the wait count for a counter kind (kMaxPosition = no wait).
  int32_t getCount(WaitCounterKind kind) const {
    return counts[static_cast<size_t>(kind)];
  }
  /// Create a WaitCnt from any wait op, reading its counts via the interface.
  /// The model determines how meta-counters (Vm/Lgkm) expand to fine-grained
  /// kinds.
  static WaitCnt fromOp(WaitCntOpInterface waitOp, WaitCounterModel model);
  /// Given a set of reaching tokens and token dependencies, compute the waited
  /// and implied tokens, the new reaching tokens after the wait, and update the
  /// wait counts so that they are consistent.
  void handleWait(ArrayRef<TokenState> reachingTokens, ValueRange dependencies,
                  SmallVectorImpl<TokenState> &waitedTokens,
                  SmallVectorImpl<TokenState> &impliedTokens,
                  SmallVectorImpl<TokenState> &nextReachingTokens,
                  llvm::function_ref<TokenState(Value)> getState,
                  WaitCounterModel model);
  /// Print the wait counts using the given counter model for formatting.
  void print(llvm::raw_ostream &os, WaitCounterModel model) const;

  bool operator==(const WaitCnt &other) const { return counts == other.counts; }

  /// Update the wait count for a counter kind (min with the existing value).
  void updateCount(WaitCounterKind kind, Position count) {
    Position &slr = counts[static_cast<size_t>(kind)];
    slr = std::min(slr, count);
  }

private:
  /// Update the wait counts from a set of tokens.
  void updateCount(ArrayRef<TokenState> tokens);
  std::array<Position, kNumKinds> counts;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const WaitCnt &cnt) {
  cnt.print(os, WaitCounterModel::GFX12_50);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitOpInfo
//===----------------------------------------------------------------------===//

/// Object for storing information about a wait operation.
struct WaitOpInfo {
  WaitOpInfo(WaitCnt counts) : counts(counts) {}
  /// Print with model-specific counter/kind names.
  void print(llvm::raw_ostream &os, WaitCounterModel model) const;
  /// Print with GFX12_50 names (for operator<<).
  void print(llvm::raw_ostream &os) const {
    print(os, WaitCounterModel::GFX12_50);
  }

  /// The computed wait counts for the wait operation.
  WaitCnt counts;
  /// The tokens that are waited on by this wait operation.
  SmallVector<TokenState> waitedTokens;
  /// The tokens that are implied (already waited on) by this wait operation.
  SmallVector<TokenState> impliedTokens;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const WaitOpInfo &info) {
  info.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitState
//===----------------------------------------------------------------------===//

/// This lattice represents the wait state at a program point.
struct WaitState : dataflow::AbstractDenseLattice {
  WaitState(LatticeAnchor anchor) : AbstractDenseLattice(anchor) {}

  /// Whether the state is empty.
  bool isEmpty() const {
    return reachingTokens.empty() && !waitOpInfo.has_value();
  }

  /// Join operation for the lattice.
  ChangeResult join(const WaitState &lattice);
  ChangeResult join(const AbstractDenseLattice &lattice) final {
    return join(static_cast<const WaitState &>(lattice));
  }
  /// Dedicated join for wait operations.
  ChangeResult joinWait(ValueRange deps, const WaitState &before,
                        WaitCnt waitCounts,
                        llvm::function_ref<TokenState(Value)> getState,
                        WaitCounterModel model);
  /// Add tokens to the reaching set. The counter model selects gfx1250
  /// (per-counter) vs CDNA (load+store and ds+km share a counter) position
  /// co-incrementing.
  ChangeResult addTokens(ArrayRef<TokenState> tokens, WaitCounterModel model);
  /// Print with model-specific counter/kind names.
  void print(raw_ostream &os, WaitCounterModel model) const;
  /// Print with GFX12_50 names (base-class override).
  void print(raw_ostream &os) const override {
    print(os, WaitCounterModel::GFX12_50);
  }

  /// Reaching tokens at this program point.
  SmallVector<TokenState> reachingTokens;
  /// Optional information about wait operations that produced this state.
  std::optional<WaitOpInfo> waitOpInfo;
};

//===----------------------------------------------------------------------===//
// WaitAnalysis
//===----------------------------------------------------------------------===//

/// A forward dataflow analysis that tracks dependency tokens produced by
/// operations implementing DependentOpInterface. The analysis computes the
/// set of pending (not-yet-waited-on) dependency tokens at each program point.
class WaitAnalysis : public dataflow::DenseForwardDataFlowAnalysis<WaitState> {
public:
  WaitAnalysis(DataFlowSolver &solver, DominanceInfo &domInfo,
               WaitCounterModel model)
      : dataflow::DenseForwardDataFlowAnalysis<WaitState>(solver),
        domInfo(domInfo), model(model) {}

  /// The wait-counter model for this analysis instance.
  WaitCounterModel getModel() const { return model; }

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op, const WaitState &before,
                               WaitState *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const WaitState &before, WaitState *after) override;

  /// Visit a call control flow transfer and update the lattice state.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const WaitState &before,
                                    WaitState *after) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const WaitState &before,
                                            WaitState *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(WaitState *lattice) override;

private:
  /// Reference to the dominance info.
  DominanceInfo &domInfo;

  /// The hardware wait-counter model (set once at construction).
  WaitCounterModel model;

  /// Map from token values to their unique IDs.
  DenseMap<Value, int32_t> tokenIDs;

  /// Temporary storage for escaped tokens during control flow transfer.
  llvm::SmallVector<TokenState> escapedTokens;

  /// Get the unique ID for a given token value.
  TokenState::ID getID(Value token) {
    auto [it, inserted] = tokenIDs.try_emplace(token, tokenIDs.size());
    assert(it->second != TokenState::kUnknownID &&
           "token ID conflicts with unknown ID");
    return it->second;
  }

  /// Get the TokenState for a given token value at a specific position.
  TokenState getState(Value token, TokenState::ID position);

  /// Map control flow operands to tokens and update the reaching and escaped
  /// tokens.
  bool mapControlFlowOperands(SmallVectorImpl<TokenState> &results,
                              SmallVectorImpl<TokenState> &scratch,
                              SmallVectorImpl<TokenState> &escapedTokens,
                              ArrayRef<TokenState> predecessorTokens,
                              ValueRange successorOperands,
                              ValueRange successorValues);
};

} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H
