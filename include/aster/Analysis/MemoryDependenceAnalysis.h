//===- MemoryDependenceAnalysis.h - Memory dependence analysis ---*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a dataflow analysis that tracks:
// 1. SSA def-use chains crossing each program point (from load operations)
// 2. Store operations whose memory locations will be read by future loads
//    (conservatively disambiguated by address SSA value + static offset)
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H
#define ASTER_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/SetVector.h"
#include <cstdint>

namespace mlir {
class Operation;
}

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// MemoryLocation - Represents a memory location for disambiguation
//===----------------------------------------------------------------------===//

/// Represents a memory location defined by an address SSA value and offset.
struct MemoryLocation {
  // The operation that accesses this memory location
  Operation *op;
  // The address register SSA value (base)
  Value address;
  // Static offset in bytes
  int64_t offset;
  // Access length in bytes (number of registers * 4)
  int64_t length;
  // Optional VGPR offset (dynamic)
  Value vgprOffset;
  // Optional SGPR offset (dynamic)
  Value sgprOffset;
  // Resource type (GlobalMemoryResource or LDSMemoryResource)
  mlir::SideEffects::Resource *resourceType;
  // If true, this location forces a flush of all pending memory operations
  bool isUniversalAlias;

  MemoryLocation(Operation *op, Value address, int64_t offset, int64_t length,
                 mlir::SideEffects::Resource *resourceType,
                 Value vgprOffset = Value(), Value sgprOffset = Value(),
                 bool isUniversalAlias = false)
      : op(op), address(address), offset(offset), length(length),
        vgprOffset(vgprOffset), sgprOffset(sgprOffset),
        resourceType(resourceType), isUniversalAlias(isUniversalAlias) {}

  /// Create a special MemoryLocation that forces flush of all pending memory
  /// ops
  static MemoryLocation
  createUniversalAlias(Operation *op,
                       mlir::SideEffects::Resource *resourceType) {
    return MemoryLocation(op, Value(), 0, 0, resourceType, Value(), Value(),
                          true);
  }

  bool operator==(const MemoryLocation &other) const {
    return op == other.op && address == other.address &&
           offset == other.offset && length == other.length &&
           vgprOffset == other.vgprOffset && sgprOffset == other.sgprOffset &&
           resourceType == other.resourceType &&
           isUniversalAlias == other.isUniversalAlias;
  }

  bool operator<(const MemoryLocation &other) const {
    if (op != other.op)
      return op < other.op;
    if (address.getAsOpaquePointer() != other.address.getAsOpaquePointer())
      return address.getAsOpaquePointer() < other.address.getAsOpaquePointer();
    if (offset != other.offset)
      return offset < other.offset;
    if (length != other.length)
      return length < other.length;
    if (vgprOffset.getAsOpaquePointer() !=
        other.vgprOffset.getAsOpaquePointer())
      return vgprOffset.getAsOpaquePointer() <
             other.vgprOffset.getAsOpaquePointer();
    if (sgprOffset.getAsOpaquePointer() !=
        other.sgprOffset.getAsOpaquePointer())
      return sgprOffset.getAsOpaquePointer() <
             other.sgprOffset.getAsOpaquePointer();
    if (resourceType != other.resourceType)
      return resourceType < other.resourceType;
    return isUniversalAlias < other.isUniversalAlias;
  }

  /// Check if this location may alias with another location.
  /// Uses range-based aliasing when possible, but conservative when dynamic
  /// offsets differ.
  ///
  /// Aliasing rules:
  /// 0. isUniversalAlias locations → ALIASES WITH EVERYTHING (same resource)
  /// 1. Different resource types → NO ALIAS
  /// 2. lsir.assume_noalias: different results from same op → NO ALIAS
  /// 3. Different base address SSA values → MAY ALIAS (conservative)
  /// 4. Different VGPR offset SSA values → MAY ALIAS (conservative)
  /// 5. Different SGPR offset SSA values → MAY ALIAS (conservative)
  /// 6. Same base address + same dynamic offsets → check byte range overlap
  ///
  /// Range overlap for case 6: [offset, offset + length) vs [other.offset,
  /// other.offset + other.length)
  bool mayAlias(const MemoryLocation &other) const;
};

} // namespace mlir::aster

// Add hash support for MemoryLocation
namespace llvm {
template <>
struct DenseMapInfo<mlir::aster::MemoryLocation> {
  using MemoryLocation = mlir::aster::MemoryLocation;

  static inline MemoryLocation getEmptyKey() {
    return MemoryLocation(
        nullptr,
        mlir::Value::getFromOpaquePointer(DenseMapInfo<void *>::getEmptyKey()),
        0, 0, nullptr, mlir::Value(), mlir::Value(), false);
  }

  static inline MemoryLocation getTombstoneKey() {
    return MemoryLocation(nullptr,
                          mlir::Value::getFromOpaquePointer(
                              DenseMapInfo<void *>::getTombstoneKey()),
                          0, 0, nullptr, mlir::Value(), mlir::Value(), false);
  }

  static unsigned getHashValue(const MemoryLocation &val) {
    return hash_combine(
        DenseMapInfo<mlir::Operation *>::getHashValue(val.op),
        DenseMapInfo<void *>::getHashValue(val.address.getAsOpaquePointer()),
        val.offset, val.length,
        DenseMapInfo<void *>::getHashValue(val.vgprOffset.getAsOpaquePointer()),
        DenseMapInfo<void *>::getHashValue(val.sgprOffset.getAsOpaquePointer()),
        val.resourceType, val.isUniversalAlias);
  }

  static bool isEqual(const MemoryLocation &lhs, const MemoryLocation &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// Template helpers for collection operations
//===----------------------------------------------------------------------===//

namespace detail {

static mlir::ChangeResult
setSetVector(llvm::SetVector<MemoryLocation> &container,
             const llvm::SetVector<MemoryLocation> &items, bool isTopState) {
  if (isTopState)
    return mlir::ChangeResult::NoChange;
  if (container == items)
    return mlir::ChangeResult::NoChange;
  container = items;
  return mlir::ChangeResult::Change;
}

static mlir::ChangeResult
appendToSetVector(llvm::SetVector<MemoryLocation> &container,
                  ArrayRef<MemoryLocation> items, bool isTopState) {
  if (isTopState)
    return mlir::ChangeResult::NoChange;
  size_t oldSize = container.size();
  container.insert_range(items);
  return container.size() != oldSize ? mlir::ChangeResult::Change
                                     : mlir::ChangeResult::NoChange;
}

static mlir::ChangeResult
eraseFromSetVector(llvm::SetVector<MemoryLocation> &container,
                   ArrayRef<MemoryLocation> items, bool isTopState) {
  if (isTopState)
    return mlir::ChangeResult::NoChange;
  size_t oldSize = container.size();
  for (const auto &item : items) {
    container.remove(item);
  }
  return container.size() != oldSize ? mlir::ChangeResult::Change
                                     : mlir::ChangeResult::NoChange;
}

// Template helper for getting collection count
static size_t
getCollectionCount(const llvm::SetVector<MemoryLocation> &container,
                   bool isTopState) {
  return isTopState ? 0 : container.size();
}
} // namespace detail

//===----------------------------------------------------------------------===//
// MemoryDependenceLattice - Lattice for memory dependence analysis
//===----------------------------------------------------------------------===//

/// This lattice represents memory dependence information at a program point.
/// It tracks:
/// 1. Pending loads that are never consumed (ordered list of loads to same
/// resource)
/// 2. Pending store operations that may be read by future loads
struct MemoryDependenceLattice : dataflow::AbstractDenseLattice {
  using PendingMemoryOpsList = llvm::SetVector<MemoryLocation>;

  MemoryDependenceLattice(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), isTopState(false),
        pendingAfterOp(PendingMemoryOpsList{}),
        mustFlushBeforeOp(PendingMemoryOpsList{}) {}

  /// Whether the state is the top state.
  bool isTop() const { return isTopState; }

  /// Whether the state is empty.
  bool isEmpty() const {
    return !isTopState && pendingAfterOp.empty() && mustFlushBeforeOp.empty();
  }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    bool changed = false;
    if (!isTop()) {
      isTopState = true;
      pendingAfterOp.clear();
      mustFlushBeforeOp.clear();
      assert(pendingAfterOp.empty() && mustFlushBeforeOp.empty() &&
             "all pending sets must be empty in top state");
      changed = true;
    }
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Meet operation for the lattice.
  ChangeResult meet(const MemoryDependenceLattice &lattice) {
    assert(false && "meet not supported atm");
    return ChangeResult::NoChange;
  }
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const MemoryDependenceLattice &>(lattice));
  }

  /// Join operation for the lattice.
  ChangeResult join(const MemoryDependenceLattice &lattice) {
    assert(false && "join not supported atm");
    return ChangeResult::NoChange;
  }
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    return join(static_cast<const MemoryDependenceLattice &>(lattice));
  }

  /// Append memory operations to the ordered set.
  ChangeResult appendPendingAfterOp(ArrayRef<MemoryLocation> ops) {
    return detail::appendToSetVector(pendingAfterOp, ops, isTopState);
  }

  /// Set memory operations (replaces existing).
  ChangeResult setPendingAfterOp(const PendingMemoryOpsList &ops) {
    return detail::setSetVector(pendingAfterOp, ops, isTopState);
  }

  /// Erase memory operations from the ordered set.
  ChangeResult erasePendingAfterOp(ArrayRef<MemoryLocation> ops) {
    return detail::eraseFromSetVector(pendingAfterOp, ops, isTopState);
  }

  /// Append memory operations to the must flush before op ordered set.
  ChangeResult appendMustFlushBeforeOp(ArrayRef<MemoryLocation> ops) {
    return detail::appendToSetVector(mustFlushBeforeOp, ops, isTopState);
  }

  /// Set must flush before op memory operations (replaces existing).
  ChangeResult setMustFlushBeforeOp(const PendingMemoryOpsList &ops) {
    return detail::setSetVector(mustFlushBeforeOp, ops, isTopState);
  }

  /// Erase memory operations from the must flush before op ordered set.
  ChangeResult eraseMustFlushBeforeOp(ArrayRef<MemoryLocation> ops) {
    return detail::eraseFromSetVector(mustFlushBeforeOp, ops, isTopState);
  }

  /// Get the pending memory operations list.
  const PendingMemoryOpsList &getPendingAfterOp() const {
    return pendingAfterOp;
  }

  /// Get the must flush before op memory operations list.
  const PendingMemoryOpsList &getMustFlushBeforeOp() const {
    return mustFlushBeforeOp;
  }

  /// Get the count of pending memory operations at this program point.
  size_t getPendingAfterOpCount() const {
    return detail::getCollectionCount(pendingAfterOp, isTopState);
  }

  /// Get the count of must flush before op memory operations at this program
  /// point.
  size_t getMustFlushBeforeOpCount() const {
    return detail::getCollectionCount(mustFlushBeforeOp, isTopState);
  }

private:
  bool isTopState;
  //
  PendingMemoryOpsList pendingAfterOp;
  // Operations that must be flushed before the current operation.
  PendingMemoryOpsList mustFlushBeforeOp;
};

//===----------------------------------------------------------------------===//
// MemoryDependenceAnalysis
//===----------------------------------------------------------------------===//

/// An analysis that, by going forward along the dataflow graph, computes
/// memory dependence information including:
/// 1. Pending loads never consumed (ordered list of loads to same resource)
/// 2. Pending stores that may be read by future loads
class MemoryDependenceAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<MemoryDependenceLattice> {
public:
  MemoryDependenceAnalysis(DataFlowSolver &solver,
                           bool flushAllMemoryOnExit = true)
      : dataflow::DenseForwardDataFlowAnalysis<MemoryDependenceLattice>(solver),
        flushAllMemoryOnExit(flushAllMemoryOnExit) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op,
                               const MemoryDependenceLattice &before,
                               MemoryDependenceLattice *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const MemoryDependenceLattice &before,
                          MemoryDependenceLattice *after) override;

  /// Visit a call control flow transfer and update the lattice state.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const MemoryDependenceLattice &before,
                                    MemoryDependenceLattice *after) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const MemoryDependenceLattice &before,
      MemoryDependenceLattice *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(MemoryDependenceLattice *lattice) override;

private:
  /// Handle propagation when either of the states are top. Returns true if
  /// either state is top.
  bool handleTopPropagation(const MemoryDependenceLattice &before,
                            MemoryDependenceLattice *after);

  /// Extract memory location from a memory operation.
  static MemoryLocation getMemoryLocation(Operation *op);

  /// Check if an operation is a load operation.
  bool isLoadOp(Operation *op);

  /// Check if an operation is a store operation.
  bool isStoreOp(Operation *op);

  /// Whether to flush all pending memory operations on end_kernel.
  bool flushAllMemoryOnExit;
};

} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::MemoryDependenceLattice)

#endif // ASTRAL_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H
