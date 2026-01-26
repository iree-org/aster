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
// This file defines a forward dataflow analysis that tracks dependency tokens
// produced by operations implementing the DependentOpInterface. The analysis
// computes, for each program point, the set of outstanding (not-yet-waited-on)
// dependency tokens.
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
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::aster::amdgcn {
class WaitOp;
class WaitAnalysis;
struct WaitState;

/// This analysis computes, for each program point, the set of outstanding
/// (not-yet-waited-on) dependency tokens. It is implemented as a dense forward
/// dataflow analysis.
///
/// The analysis tracks dependency tokens produced by operations implementing
/// DependentOpInterface (load/store operations) and consumed via WaitOps. Key
/// capabilities include:
///
/// - Token production tracking from load/store operations
/// - Token consumption via WaitOps with position-based counting
/// - Position updates as new tokens are pushed (stack-like ordering for FIFO
///   hardware counters)
/// - Control flow merging with dominance-aware token filtering
/// - Region-based control flow transfers
///
/// The analysis is built on several core components:
///
/// - TokenState: Represents a dependency token with its ID, position, and
///   token kind.
///
/// - WaitCnt: Helper struct for managing hardware wait count values (vm_cnt,
///   lgkm_cnt). Provides utilities for computing counts from WaitOps and
///   updating counts based on token positions.
///
/// - WaitState: Lattice element tracking reaching tokens at each program point,
///   plus optional WaitOpInfo that captures waited tokens, implied tokens, and
///   computed wait counts.

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
  TokenState(Value token, ID id, MemoryInstructionKind kind,
             Position position = 0)
      : token(token), id(id), position(position), kind(kind) {}

  /// Create an unknown token state for dmem.
  static TokenState unknownDMem(Position pos) {
    return TokenState(nullptr, kUnknownID, MemoryInstructionKind::Shared, pos);
  }
  /// Create an unknown token state for smem.
  static TokenState unknownSMem(Position pos) {
    return TokenState(nullptr, kUnknownID, MemoryInstructionKind::Constant,
                      pos);
  }
  /// Create an unknown token state for dmem.
  static TokenState unknownVMem(Position pos) {
    return TokenState(nullptr, kUnknownID, MemoryInstructionKind::Flat, pos);
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
  /// Returns the memory instruction kind of the token.
  MemoryInstructionKind getKind() const { return kind; }
  /// Print the state.
  void print(raw_ostream &os) const;

private:
  Value token = nullptr;
  ID id = -1;
  Position position = 0;
  MemoryInstructionKind kind = MemoryInstructionKind::None;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const TokenState &state) {
  state.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

/// Object representing wait counts. Note that this class models the semantics
/// of CDNA3 and CDNA4. TODO: Expand to other architectures.
struct WaitCnt {
  using Position = TokenState::Position;
  static constexpr Position kMaxPosition = TokenState::kMaxPosition;
  WaitCnt(Position vmCnt = kMaxPosition, Position lgkmCnt = kMaxPosition)
      : vmCnt(vmCnt), lgkmCnt(lgkmCnt) {}

  /// Returns the vm wait count.
  int32_t getVmcnt() const { return vmCnt; }
  /// Returns the lgkm wait count.
  int32_t getLgkmcnt() const { return lgkmCnt; }
  /// Create WaitCnt object from a WaitOp.
  static WaitCnt fromOp(WaitOp waitOp);
  /// Get the wait count for a given memory instruction kind.
  int32_t getCount(MemoryInstructionKind kind) const;
  /// Given a set of reaching tokens and token dependencies, compute the waited
  /// and implied tokens, the new reaching tokens after the wait, and update the
  /// wait counts so that they are consistent.
  void handleWait(ArrayRef<TokenState> reachingTokens, ValueRange dependencies,
                  SmallVectorImpl<TokenState> &waitedTokens,
                  SmallVectorImpl<TokenState> &impliedTokens,
                  SmallVectorImpl<TokenState> &nextReachingTokens,
                  llvm::function_ref<TokenState(Value)> getState);

  /// Handle escaped tokens by converting them to unknown tokens at their
  /// dominant positions, and merging them into the results.
  static bool handleEscapedTokens(SmallVectorImpl<TokenState> &results,
                                  SmallVectorImpl<TokenState> &escapedTokens);

  /// Print the wait counts.
  void print(llvm::raw_ostream &os) const;

  bool operator==(const WaitCnt &other) const {
    return vmCnt == other.vmCnt && lgkmCnt == other.lgkmCnt;
  }

private:
  /// Update the wait count with the given kind and position.
  void updateCount(MemoryInstructionKind kind, Position count);
  /// Update the wait count from a set of tokens.
  void updateCount(ArrayRef<TokenState> tokens);
  Position vmCnt;
  Position lgkmCnt;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const WaitCnt &cnt) {
  cnt.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitOpInfo
//===----------------------------------------------------------------------===//

/// Object for storing information about a wait operation.
struct WaitOpInfo {
  WaitOpInfo(WaitCnt counts) : counts(counts) {}
  /// Print the wait op info.
  void print(llvm::raw_ostream &os) const;

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
                        llvm::function_ref<TokenState(Value)> getState);
  /// Add tokens to the reaching set.
  ChangeResult addTokens(ArrayRef<TokenState> tokens);
  /// Print the lattice element.
  void print(raw_ostream &os) const override;

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
  using dataflow::DenseForwardDataFlowAnalysis<
      WaitState>::DenseForwardDataFlowAnalysis;
  WaitAnalysis(DataFlowSolver &solver, DominanceInfo &domInfo)
      : dataflow::DenseForwardDataFlowAnalysis<WaitState>(solver),
        domInfo(domInfo) {}

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
  /// Handle a WaitOp operation.
  LogicalResult handleWaitOp(WaitOp waitOp, const WaitState &before,
                             WaitState *after);

  /// Handle an operation.
  LogicalResult handleOp(Operation *op, const WaitState &before,
                         WaitState *after);
  /// Reference to the dominance info.
  DominanceInfo &domInfo;

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

  /// Control flow transfer function. This function propagates tokens from
  /// predecessor to successor, taking into account dominance, as well as the
  /// tokens being forwarded by control-flow.
  bool getReachingTokens(SmallVectorImpl<TokenState> &results,
                         SmallVectorImpl<TokenState> &scratch,
                         SmallVectorImpl<TokenState> &escapedTokens,
                         ArrayRef<TokenState> predecessorTokens,
                         ValueRange successorOperands, Block *successor);
  bool getReachingTokens(SmallVectorImpl<TokenState> &results,
                         SmallVectorImpl<TokenState> &scratch,
                         SmallVectorImpl<TokenState> &escapedTokens,
                         ArrayRef<TokenState> predecessorTokens,
                         RegionBranchOpInterface op, RegionBranchPoint point,
                         RegionSuccessor successor);
};

} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H
