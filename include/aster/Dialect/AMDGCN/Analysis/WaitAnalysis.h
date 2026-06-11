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
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <variant>

namespace mlir::aster::amdgcn {
enum class ISAVersion : uint32_t;
class WaitOp;
class WaitCntOpInterface;
class WaitAnalysisBase;
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

/// How a wait counter retires the operations it tracks.
/// - InOrder: completions in program order, a wait can use exact partial count.
/// - OutOfOrder: completions may be reordered (e.g. scalar memory), a wait
///   cannot trust a partial count and must flush the counter to zero.
/// Which counter kinds are out of order is hardware generation-specific.
enum class DrainBehavior : uint8_t {
  InOrder,
  OutOfOrder,
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
  /// Print the state.
  void print(raw_ostream &os) const;

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

/// Decode a dependency token's memory class from its token type, or nullopt
/// if the value is not a dependency token. Set `isWrite` for write tokens.
inline std::optional<MemoryInstructionKind> tokenMemoryKind(Value token,
                                                            bool &isWrite) {
  Type type = token.getType();
  if (auto readToken = dyn_cast<ReadTokenType>(type)) {
    isWrite = false;
    return readToken.getKind();
  }
  if (auto writeToken = dyn_cast<WriteTokenType>(type)) {
    isWrite = true;
    return writeToken.getKind();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

/// CDNA3/CDNA4 wait counts:
///   - vmcnt for vmem load+store
///   - lgkmcnt for LDS (ds) and scalar (km)
struct WaitCntCdna3 {
  using Position = TokenState::Position;
  static constexpr Position kMaxPosition = TokenState::kMaxPosition;

  Position vmcnt = kMaxPosition;
  Position lgkmcnt = kMaxPosition;

  /// The wait count tracking `kind` (kMaxPosition = no wait).
  int32_t getCount(WaitCounterKind kind) const {
    return isVmCounter(kind) ? vmcnt : lgkmcnt;
  }
  /// Min-update the counter tracking `kind`.
  void updateCount(WaitCounterKind kind, Position count) {
    Position &c = isVmCounter(kind) ? vmcnt : lgkmcnt;
    c = std::min(c, count);
  }
  void print(llvm::raw_ostream &os) const;
  bool operator==(const WaitCntCdna3 &other) const {
    return vmcnt == other.vmcnt && lgkmcnt == other.lgkmcnt;
  }

private:
  // vmem (Vm) -> vmcnt; ds (LDS), scalar_read/km (scalar), and the lgkm drain
  // kind -> lgkmcnt.
  static bool isVmCounter(WaitCounterKind kind) {
    assert((kind == WaitCounterKind::Vm || kind == WaitCounterKind::Ds ||
            kind == WaitCounterKind::Km ||
            kind == WaitCounterKind::ScalarRead ||
            kind == WaitCounterKind::Lgkm) &&
           "not a CDNA counter kind");
    return kind == WaitCounterKind::Vm;
  }
};

/// gfx1250 wait counts: one independent counter per kind
/// {load, store, ds, km, tensor}.
struct WaitCntGfx1250 {
  using Position = TokenState::Position;
  static constexpr Position kMaxPosition = TokenState::kMaxPosition;

  /// The counter kinds gfx1250 tracks, in print/iteration order.
  static constexpr std::array<WaitCounterKind, 5> kKinds = {
      WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds,
      WaitCounterKind::Km, WaitCounterKind::Tensor};

  Position loadcnt = kMaxPosition;
  Position storecnt = kMaxPosition;
  Position dscnt = kMaxPosition;
  Position kmcnt = kMaxPosition;
  Position tensorcnt = kMaxPosition;

  int32_t getCount(WaitCounterKind kind) const {
    return const_cast<WaitCntGfx1250 *>(this)->counterFor(kind);
  }
  /// Min-update the counter tracking `kind`.
  void updateCount(WaitCounterKind kind, Position count) {
    Position &c = counterFor(kind);
    c = std::min(c, count);
  }
  void print(llvm::raw_ostream &os) const;
  bool operator==(const WaitCntGfx1250 &other) const {
    return loadcnt == other.loadcnt && storecnt == other.storecnt &&
           dscnt == other.dscnt && kmcnt == other.kmcnt &&
           tensorcnt == other.tensorcnt;
  }

private:
  /// Select the field tracking `kind`.
  Position &counterFor(WaitCounterKind kind) {
    assert(kind == WaitCounterKind::Load || kind == WaitCounterKind::Store ||
           kind == WaitCounterKind::Ds || kind == WaitCounterKind::Km ||
           kind == WaitCounterKind::ScalarRead ||
           kind == WaitCounterKind::Tensor && "expected gfx1250 counter");
    switch (kind) {
    case WaitCounterKind::Load:
      return loadcnt;
    case WaitCounterKind::Store:
      return storecnt;
    case WaitCounterKind::Ds:
      return dscnt;
    case WaitCounterKind::Km:
    case WaitCounterKind::ScalarRead:
      return kmcnt;
    case WaitCounterKind::Tensor:
      return tensorcnt;
    default:
      llvm_unreachable("counter kind is not a gfx1250 counter");
    }
  }
};

/// The wait counts of a wait op: various HW iterations have different counter
/// sets, they each get their own struct.
using WaitCnt = std::variant<WaitCntCdna3, WaitCntGfx1250>;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const WaitCnt &cnt) {
  std::visit([&](const auto &c) { c.print(os); }, cnt);
  return os;
}

//===----------------------------------------------------------------------===//
// WaitOpInfo
//===----------------------------------------------------------------------===//

/// Object for storing information about a wait operation.
struct WaitOpInfo {
  WaitOpInfo(WaitCnt counts) : counts(counts) {}

  /// Print the wait info.
  void print(llvm::raw_ostream &os) const;

  /// Content equality, used for dataflow change detection (compares counts and
  /// both token lists, not just their sizes).
  bool operator==(const WaitOpInfo &other) const {
    return counts == other.counts && waitedTokens == other.waitedTokens &&
           impliedTokens == other.impliedTokens;
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

  /// Join operation for the lattice (control-flow merge of reaching tokens).
  ChangeResult join(const WaitState &lattice);
  ChangeResult join(const AbstractDenseLattice &lattice) final {
    return join(static_cast<const WaitState &>(lattice));
  }
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
class WaitAnalysisBase
    : public dataflow::DenseForwardDataFlowAnalysis<WaitState> {
public:
  WaitAnalysisBase(DataFlowSolver &solver, DominanceInfo &domInfo)
      : dataflow::DenseForwardDataFlowAnalysis<WaitState>(solver),
        domInfo(domInfo) {}

  /// Read the wait counts a wait op requests via the interface.
  virtual WaitCnt countsFromWaitOp(WaitCntOpInterface waitOp) const = 0;

  /// Add `produced` tokens to `state`, advancing the position of every already-
  /// reaching token that shares a physical counter with a produced token.
  /// Returns whether `state` changed.
  virtual ChangeResult advanceAndAdd(WaitState &state,
                                     ArrayRef<TokenState> produced) const = 0;

  /// Apply a wait op: from `before`, the op's requested `counts`, and its
  /// `deps`, compute the waited/implied tokens and the surviving reaching set
  /// into `after`. Returns whether `after` changed.
  virtual ChangeResult
  transferWait(const WaitState &before, ValueRange deps, WaitCnt counts,
               WaitState &after,
               llvm::function_ref<TokenState(Value)> getState) const = 0;

  /// Map a dependency token to the wait-counter kind it should be tracked
  /// under, or nullopt if `token` is not a dependency token. The mapping is
  /// WaitCounterModel-specific: the same op can complete on different physical
  /// counters across hardware generations (e.g. a global_load_dword is a `vmem`
  /// counter on CDNA but a `load` counter on gfx1250).
  virtual std::optional<WaitCounterKind> getCounterKind(Value token) const = 0;

  /// The drain behavior of a counter kind on this hardware (the per-hardware
  /// kind -> DrainBehavior map). A wait that may depend on an OutOfOrder kind
  /// must flush that kind's counter to zero rather than use a partial count.
  virtual DrainBehavior drainBehavior(WaitCounterKind kind) const = 0;

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

protected:
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

  /// Map control flow operands to tokens and update the reaching and escaped
  /// tokens.
  bool mapControlFlowOperands(SmallVectorImpl<TokenState> &results,
                              SmallVectorImpl<TokenState> &scratch,
                              SmallVectorImpl<TokenState> &escapedTokens,
                              ArrayRef<TokenState> predecessorTokens,
                              ValueRange successorOperands,
                              ValueRange successorValues);
};

/// CDNA3/CDNA4 wait analysis: vmcnt + lgkmcnt.
class WaitAnalysisCdna : public WaitAnalysisBase {
public:
  using WaitAnalysisBase::WaitAnalysisBase;
  WaitCnt countsFromWaitOp(WaitCntOpInterface waitOp) const override;
  ChangeResult advanceAndAdd(WaitState &state,
                             ArrayRef<TokenState> produced) const override;
  ChangeResult
  transferWait(const WaitState &before, ValueRange deps, WaitCnt counts,
               WaitState &after,
               llvm::function_ref<TokenState(Value)> getState) const override;

  // Note: CDNA3/4 has only vm, lgkm and exp counters but because scalar memory
  // reads are out of order, we need to track them separately from the rest.
  // Note: exec counter tracking + lowering is NYI beyond getCounterKind.
  std::optional<WaitCounterKind> getCounterKind(Value token) const override {
    bool isWrite = false;
    std::optional<MemoryInstructionKind> kind = tokenMemoryKind(token, isWrite);
    if (!kind)
      return std::nullopt;
    switch (*kind) {
    case MemoryInstructionKind::Flat:
      // vmem load+store: one kind (CDNA has a single vmcnt for both).
      return WaitCounterKind::Vm;
    case MemoryInstructionKind::Shared:
      // LDS: in order on lgkmcnt, so partial waits are exact.
      return WaitCounterKind::Lgkm;
    case MemoryInstructionKind::Constant:
      // Scalar reads are out of order on lgkmcnt and force a full lgkm drain.
      return isWrite ? WaitCounterKind::Lgkm : WaitCounterKind::ScalarRead;
    case MemoryInstructionKind::Exec:
      return WaitCounterKind::Exec;
    default:
      // CDNA3/CDNA4 have no tensor counter.
      return std::nullopt;
    }
  }

  DrainBehavior drainBehavior(WaitCounterKind kind) const override {
    // CDNA3/4 manual 4.4:
    // ... except scalar-memory reads, which can return out-of order (in which
    // case S_WAITCNT 0 is the only legitimate value).
    if (kind == WaitCounterKind::ScalarRead)
      return DrainBehavior::OutOfOrder;
    return DrainBehavior::InOrder;
  }
};

/// gfx1250 wait analysis: load, store, ds, km, tensor, exec, async counters.
class WaitAnalysisGfx1250 : public WaitAnalysisBase {
public:
  using WaitAnalysisBase::WaitAnalysisBase;
  WaitCnt countsFromWaitOp(WaitCntOpInterface waitOp) const override;
  ChangeResult advanceAndAdd(WaitState &state,
                             ArrayRef<TokenState> produced) const override;
  ChangeResult
  transferWait(const WaitState &before, ValueRange deps, WaitCnt counts,
               WaitState &after,
               llvm::function_ref<TokenState(Value)> getState) const override;

  // Note: gfx1250 splits CDNA's lgkmcnt into independent counters; ASTER models
  // load, store, ds, km, scalar_read, tensor, exec and async. Scalar memory
  // reads (scalar_read) return out of order and force a full km_cnt drain.
  // Note: exec/async counter tracking + lowering are NYI beyond getCounterKind.
  std::optional<WaitCounterKind> getCounterKind(Value token) const override {
    bool isWrite = false;
    std::optional<MemoryInstructionKind> kind = tokenMemoryKind(token, isWrite);
    if (!kind)
      return std::nullopt;
    switch (*kind) {
    case MemoryInstructionKind::Flat:
      return isWrite ? WaitCounterKind::Store : WaitCounterKind::Load;
    case MemoryInstructionKind::Shared:
      return WaitCounterKind::Ds;
    case MemoryInstructionKind::Constant:
      return isWrite ? WaitCounterKind::Km : WaitCounterKind::ScalarRead;
    case MemoryInstructionKind::Exec:
      return WaitCounterKind::Exec;
    case MemoryInstructionKind::Tensor:
      return WaitCounterKind::Tensor;
    case MemoryInstructionKind::Async:
      return WaitCounterKind::Async;
    default:
      return std::nullopt;
    }
  }

  /// TODO: Model this properly when ISA manual is available, for now use
  /// CDNA3/4 rules.
  DrainBehavior drainBehavior(WaitCounterKind kind) const override {
    // CDNA3/4 manual 4.4:
    // ... except scalar - memoryreads, which can return out-of order (in which
    // case S_WAITCNT 0 is the only legitimate value).
    if (kind == WaitCounterKind::ScalarRead)
      return DrainBehavior::OutOfOrder;
    return DrainBehavior::InOrder;
  }
};

/// Load the wait analysis for `model` into `solver` and return it.
/// Single entry-point to select the implementation by WaitCounterModel.
WaitAnalysisBase &loadWaitAnalysis(DataFlowSolver &solver,
                                   DominanceInfo &domInfo,
                                   WaitCounterModel model);

} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_WAITANALYSIS_H
