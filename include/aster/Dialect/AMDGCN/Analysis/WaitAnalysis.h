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
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::aster::amdgcn {
class WaitOp;
class WaitAnalysis;
struct WaitState;

/// This analysis computes, for each program point, the set of outstanding
/// (not-yet-waited-on) dependency tokens. It is implemented as a dense forward
/// dataflow analysis using MLIR's DataFlowFramework.
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
///   MemoryInstructionKind (flat, constant, shared). Includes factory methods
///   for unknown escaped tokens and merge semantics for join points.
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
  /// Create an unknown token state for vmem.
  static TokenState unknownVMem(Position pos) {
    return TokenState(nullptr, kUnknownID, MemoryInstructionKind::Flat, pos);
  }

  bool operator==(Value value) const { return token == value; }
  bool operator==(const TokenState &other) const {
    return token == other.token && kind == other.kind && id == other.id &&
           position == other.position;
  }
  bool operator<(const TokenState &other) const {
    if (kind != other.kind)
      return kind < other.kind;
    return id < other.id;
  }

  TokenState &operator++();
  bool merge(const TokenState &other);
  void print(raw_ostream &os) const;

  Value getToken() const { return token; }
  ID getID() const { return id; }
  Position getPosition() const { return position; }
  MemoryInstructionKind getKind() const { return kind; }

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
  static constexpr Position kMaxPosition = std::numeric_limits<Position>::max();

  WaitCnt(Position vmCnt = kMaxPosition, Position lgkmCnt = kMaxPosition)
      : vmCnt(vmCnt), lgkmCnt(lgkmCnt) {}

  static WaitCnt fromOp(WaitOp waitOp);
  void setOpCounts(WaitOp waitOp) const;
  int32_t getCount(MemoryInstructionKind kind) const;
  void updateCount(MemoryInstructionKind kind, Position count);
  void updateCount(ArrayRef<TokenState> tokens);
  void print(llvm::raw_ostream &os) const;

  /// Given a set of reaching tokens and token dependencies, compute the waited
  /// and implied tokens, the new reaching tokens after the wait, and update the
  /// wait counts so that they are consistent.
  void handleWait(ArrayRef<TokenState> reachingTokens, ValueRange dependencies,
                  SmallVectorImpl<TokenState> &waitedTokens,
                  SmallVectorImpl<TokenState> &impliedTokens,
                  SmallVectorImpl<TokenState> &nextReachingTokens,
                  llvm::function_ref<TokenState(Value)> getState);

  static bool handleEscapedTokens(SmallVectorImpl<TokenState> &results,
                                  SmallVectorImpl<TokenState> &escapedTokens);

  bool operator==(const WaitCnt &other) const {
    return vmCnt == other.vmCnt && lgkmCnt == other.lgkmCnt;
  }

  Position vmCnt = kMaxPosition;
  Position lgkmCnt = kMaxPosition;
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
  void print(llvm::raw_ostream &os) const;

  WaitCnt counts;
  SmallVector<TokenState> waitedTokens;
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

  bool isEmpty() const {
    return reachingTokens.empty() && !waitOpInfo.has_value();
  }

  ChangeResult join(const WaitState &lattice);
  ChangeResult join(const AbstractDenseLattice &lattice) final {
    return join(static_cast<const WaitState &>(lattice));
  }
  ChangeResult joinWait(ValueRange deps, const WaitState &before,
                        WaitCnt waitCounts,
                        llvm::function_ref<TokenState(Value)> getState);
  ChangeResult addTokens(ArrayRef<TokenState> tokens);
  void print(raw_ostream &os) const override;

  SmallVector<TokenState> reachingTokens;
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

  LogicalResult visitOperation(Operation *op, const WaitState &before,
                               WaitState *after) override;
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const WaitState &before, WaitState *after) override;
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const WaitState &before,
                                    WaitState *after) override;
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const WaitState &before,
                                            WaitState *after) override;
  void setToEntryState(WaitState *lattice) override;

  LogicalResult handleWaitOp(WaitOp waitOp, const WaitState &before,
                             WaitState *after);
  LogicalResult handleOp(Operation *op, const WaitState &before,
                         WaitState *after);

  static MemoryInstructionKind getMemoryKindFromToken(Value token);
  TokenState createTokenState(Value token);

  bool filterByDominance(SmallVectorImpl<TokenState> &results,
                         SmallVectorImpl<TokenState> &scratch,
                         SmallVectorImpl<TokenState> &escapedTokens,
                         ArrayRef<TokenState> predecessorTokens,
                         llvm::function_ref<bool(Value)> dominates);

  bool mapControlFlowOperands(SmallVectorImpl<TokenState> &results,
                              SmallVectorImpl<TokenState> &scratch,
                              SmallVectorImpl<TokenState> &escapedTokens,
                              ArrayRef<TokenState> predecessorTokens,
                              ValueRange operands, ValueRange successorValues);

  DominanceInfo &domInfo;
  DenseMap<Value, int32_t> tokenIDs;
  SmallVector<TokenState> escapedTokens;
};

} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H
