//===- MemoryDependenceAnalysis.cpp - Memory dependence analysis ---------===//
//
// Copyright 2025 The ASTRAL Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/ValueOrConst.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

#define DEBUG_TYPE "memory-dependence-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::lsir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// MemoryDependenceLattice
//===----------------------------------------------------------------------===//

void MemoryDependenceLattice::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<top>";
    return;
  }

  os << "{ pending memory ops: ";
  if (isEmpty() || pendingAfterOp.empty()) {
    os << "[]";
  } else {
    const PendingMemoryOpsList &ops = getPendingAfterOp();
    os << "[";
    llvm::interleaveComma(
        ops, os, [&](const MemoryLocation &loc) { os << loc.op->getName(); });
    os << "]";
  }
  os << "\n\t\tmust flush before op: ";
  if (isEmpty() || mustFlushBeforeOp.empty()) {
    os << "[]";
  } else {
    const PendingMemoryOpsList &ops = getMustFlushBeforeOp();
    os << "[";
    llvm::interleaveComma(
        ops, os, [&](const MemoryLocation &loc) { os << loc.op->getName(); });
    os << "]";
  }
  os << " }";
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::MemoryDependenceLattice)

//===----------------------------------------------------------------------===//
// MemoryLocation - Implementation
//===----------------------------------------------------------------------===//

bool MemoryLocation::mayAlias(const MemoryLocation &other) const {
  // Universal alias locations alias with everything of the same resource type
  if (isUniversalAlias || other.isUniversalAlias) {
    return resourceType == other.resourceType;
  }

  // Different resource types never alias
  if (resourceType != other.resourceType)
    return false;

  // Check if both addresses come from different results of the same
  // lsir.assume_noalias op
  if (address && other.address) {
    Operation *thisDefOp = address.getDefiningOp();
    Operation *otherDefOp = other.address.getDefiningOp();

    if (thisDefOp && otherDefOp && thisDefOp == otherDefOp) {
      if (isa<lsir::AssumeNoaliasOp>(thisDefOp)) {
        // Both addresses come from the same lsir.assume_noalias op
        // Check if they are different results
        auto thisResult = dyn_cast<OpResult>(address);
        auto otherResult = dyn_cast<OpResult>(other.address);
        if (thisResult && otherResult &&
            thisResult.getResultNumber() != otherResult.getResultNumber()) {
          // Different results from the same lsir.assume_noalias op don't alias
          LDBG() << "No alias: different results of lsir.assume_noalias";
          return false;
        }
      }
    }
  }

  // Different base address SSA values may alias (conservative)
  if (address != other.address)
    return true;

  // Different VGPR offset SSA values may alias (conservative)
  // We can only be precise if both have no VGPR offset OR the same VGPR
  // offset value
  if (vgprOffset != other.vgprOffset)
    return true;

  // Different SGPR offset SSA values may alias (conservative)
  // We can only be precise if both have no SGPR offset OR the same SGPR
  // offset value
  if (sgprOffset != other.sgprOffset)
    return true;

  // Same base address and same dynamic offsets: check if byte ranges overlap
  // The effective address is: base + static_offset + vgpr_offset +
  // sgpr_offset Since the dynamic parts are the same, we can compare just the
  // static parts Two ranges [a, a + len_a) and [b, b + len_b) overlap if:
  //   a < b + len_b  AND  b < a + len_a
  int64_t thisEnd = offset + length;
  int64_t otherEnd = other.offset + other.length;

  return (offset < otherEnd) && (other.offset < thisEnd);
}

//===----------------------------------------------------------------------===//
// MemoryDependenceAnalysis - Helper functions
//===----------------------------------------------------------------------===//

/// Check if an operation has a memory effect on the given resource type
template <typename EffectType, typename ResourceType>
static bool hasMemoryEffect(Operation &op) {
  auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(&op);
  if (!memoryEffects)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  memoryEffects.getEffects(effects);

  auto *resource = ResourceType::get();

  return llvm::any_of(effects, [&](const auto &effect) {
    return isa<EffectType>(effect.getEffect()) &&
           effect.getResource() == resource;
  });
}

bool MemoryDependenceAnalysis::isLoadOp(Operation *op) {
  return hasMemoryEffect<MemoryEffects::Read, GlobalMemoryResource>(*op) ||
         hasMemoryEffect<MemoryEffects::Read, LDSMemoryResource>(*op);
}

bool MemoryDependenceAnalysis::isStoreOp(Operation *op) {
  return hasMemoryEffect<MemoryEffects::Write, GlobalMemoryResource>(*op) ||
         hasMemoryEffect<MemoryEffects::Write, LDSMemoryResource>(*op);
}

static int64_t computeAccessLength(Type type) {
  if (auto regType = dyn_cast<RegisterTypeInterface>(type)) {
    std::optional<int64_t> sizeInBytes = regType.getSizeInBytes();
    assert(sizeInBytes.has_value() && "register type must have valid size");
    return *sizeInBytes;
  }
  // Conservative: assume 4 bytes if we can't determine precisely.
  return 4;
}

/// TODO: extract in some memory access interface
/// Helper to extract memory location from memory operations.
MemoryLocation MemoryDependenceAnalysis::getMemoryLocation(Operation *op) {
  Value address;
  Value dataValue;
  Value vgprOffset;
  Value sgprOffset;
  int64_t offset = 0; // Default static offset
  int64_t length = 4; // Default to 4 bytes (1 register)
  mlir::SideEffects::Resource *resourceType = nullptr;

  // Determine the resource type
  if (hasMemoryEffect<MemoryEffects::Read, GlobalMemoryResource>(*op) ||
      hasMemoryEffect<MemoryEffects::Write, GlobalMemoryResource>(*op)) {
    resourceType = GlobalMemoryResource::get();
  } else if (hasMemoryEffect<MemoryEffects::Read, LDSMemoryResource>(*op) ||
             hasMemoryEffect<MemoryEffects::Write, LDSMemoryResource>(*op)) {
    resourceType = LDSMemoryResource::get();
  } else {
    llvm_unreachable("Unknown memory operation");
  }

  if (auto load = dyn_cast<amdgcn::LoadOp>(op)) {
    address = load.getAddr();
    dataValue = load.getDest();
    vgprOffset = load.getDynamicOffset();
    sgprOffset = load.getUniformOffset();
    // FIXME: This doesn't handle unrealized constant offsets.
    offset =
        !load.getConstantOffset()
            ? 0
            : cast<ValueOrI32>(load.getConstantOffset()).getConst().value_or(0);
  } else if (auto store = dyn_cast<amdgcn::StoreOp>(op)) {
    address = store.getAddr();
    dataValue = store.getData();
    vgprOffset = store.getDynamicOffset();
    sgprOffset = store.getUniformOffset();
    // FIXME: This doesn't handle unrealized constant offsets.
    offset = !store.getConstantOffset()
                 ? 0
                 : cast<ValueOrI32>(store.getConstantOffset())
                       .getConst()
                       .value_or(0);
  } else {
    llvm_unreachable("Unknown memory operation");
  }

  assert(address && "Address is required for memory operations");

  // Compute the access length from the data type
  if (dataValue)
    length = computeAccessLength(dataValue.getType());

  return MemoryLocation(op, address, offset, length, resourceType, vgprOffset,
                        sgprOffset);
}

bool MemoryDependenceAnalysis::handleTopPropagation(
    const MemoryDependenceLattice &before, MemoryDependenceLattice *after) {
  if (before.isTop() || after->isTop()) {
    propagateIfChanged(after, after->setToTop());
    return true;
  }
  return false;
}

#define DUMP_STATE_HELPER(name, obj)                                           \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "Visiting " name ": " << obj << "\n";                              \
      os << "  Incoming lattice: ";                                            \
      before.print(os);                                                        \
      os << "\n  Outgoing lattice: ";                                          \
      after->print(os);                                                        \
    });                                                                        \
  });

//===----------------------------------------------------------------------===//
// MemoryDependenceAnalysis - Helper functions
//===----------------------------------------------------------------------===//

/// Flush pending memory operations from the beginning up to and including
/// the first matching operation. The predicate should return true when a match
/// is found. Only memory ops on the same resource type are flushed.
/// Returns true if a match was found and operations were flushed.
static void flushPendingMemoryOpsUpToMatch(
    const MemoryDependenceLattice &before, MemoryDependenceLattice *after,
    mlir::SideEffects::Resource *resourceType,
    llvm::function_ref<bool(const MemoryLocation &)> predicate,
    ChangeResult &result) {
  const auto &pendingAfterOp = before.getPendingAfterOp();
  int ridx = 0;
  for (auto it = pendingAfterOp.rbegin(); it != pendingAfterOp.rend();
       ++it, ++ridx) {
    const MemoryLocation &memLoc = *it;
    if (!predicate(memLoc))
      continue;

    int idx = pendingAfterOp.size() - ridx;
    SmallVector<MemoryLocation> opsToFlush;
    for (int i = 0; i < idx; ++i) {
      if (memLoc.resourceType != pendingAfterOp[i].resourceType)
        continue;
      opsToFlush.push_back(pendingAfterOp[i]);
    }
    result |= after->appendMustFlushBeforeOp(opsToFlush);
    result |= after->erasePendingAfterOp(opsToFlush);
    return;
  }
}

//===----------------------------------------------------------------------===//
// MemoryDependenceAnalysis - Visit methods
//===----------------------------------------------------------------------===//

// Forward analysis: We're going from before to after
// - killedLoads: pending loads that must be consumed by this operation
// - newLoads: pending loads created by this operation
// - killedStores: stores that are consumed by this operation (read by a load)
// - newStores: stores created by this operation (become pending)
LogicalResult
MemoryDependenceAnalysis::visitOperation(Operation *op,
                                         const MemoryDependenceLattice &before,
                                         MemoryDependenceLattice *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));

  if (handleTopPropagation(before, after))
    return success();

  ChangeResult result = ChangeResult::NoChange;
  result |= after->setPendingAfterOp(before.getPendingAfterOp());

  // Handle end_kernel as universal alias - flush all pending memory operations
  // (if enabled via flushAllMemoryOnExit flag)
  if (isa<amdgcn::EndKernelOp>(op)) {
    if (flushAllMemoryOnExit) {
      // Flush all pending operations for both GlobalMemoryResource and
      // LDSMemoryResource
      for (auto *resourceType : std::array<mlir::SideEffects::Resource *, 2>{
               GlobalMemoryResource::get(), LDSMemoryResource::get()}) {
        const auto &pendingAfterOp = before.getPendingAfterOp();
        SmallVector<MemoryLocation> opsToFlush;
        for (const auto &memLoc : pendingAfterOp) {
          if (memLoc.resourceType == resourceType) {
            opsToFlush.push_back(memLoc);
          }
        }
        if (!opsToFlush.empty()) {
          result |= after->appendMustFlushBeforeOp(opsToFlush);
          result |= after->erasePendingAfterOp(opsToFlush);
        }
      }
    }
    propagateIfChanged(after, result);
    return success();
  }

  if (isLoadOp(op) || isStoreOp(op)) {
    auto loc = getMemoryLocation(op);
    result |= after->appendPendingAfterOp(loc);
  }

  // Note: be sure to run DCE to avoid dead loads that over-synchronize
  bool isLoad = isLoadOp(op);
  bool isStore = isStoreOp(op);
  if (isLoad || isStore) {
    auto loc = getMemoryLocation(op);

    std::function<bool(const MemoryLocation &)> loadPredicate =
        [&](const MemoryLocation &memLoc) {
          bool res = isStoreOp(memLoc.op) && loc.mayAlias(memLoc);
          if (res)
            LDBG() << "predicate: " << *memLoc.op << " ALIASES with "
                   << *loc.op;
          else
            LDBG() << "predicate: " << *memLoc.op << " DOES NOT alias "
                   << *loc.op;
          return res;
        };
    std::function<bool(const MemoryLocation &)> storePredicate =
        [&](const MemoryLocation &memLoc) {
          return isLoadOp(memLoc.op) && loc.mayAlias(memLoc);
        };

    // Load must flush any previous pending memory stores issued before that may
    // alias. Store must flush any previous pending memory loads issued before
    // that may alias.
    // Given the waitcnt behavior on AMDGCN, when we hit the first memory
    // load that aliases, we need to flush all the pending memory in [begin, it]
    // on the same resource type.
    flushPendingMemoryOpsUpToMatch(before, after, loc.resourceType,
                                   isLoad ? loadPredicate : storePredicate,
                                   result);

    result |= after->appendPendingAfterOp(loc);
  }

  // Examine any operand that comes from a load.
  // When a load's result is used, we need to flush all pending memory ops
  // from [begin, it] where it is the position of this load. Given the waitcnt
  // behavior on AMDGCN, when we hit the first load that is consumed, we need
  // to flush all the ops in [begin, it].
  for (Value operand : op->getOperands()) {
    Operation *loadOp = operand.getDefiningOp();
    if (!isLoadOp(loadOp))
      continue;

    auto loadLoc = getMemoryLocation(loadOp);
    flushPendingMemoryOpsUpToMatch(
        before, after, loadLoc.resourceType,
        [&](const MemoryLocation &memLoc) { return memLoc == loadLoc; },
        result);
  }

  propagateIfChanged(after, result);
  return success();
}

void MemoryDependenceAnalysis::visitBlockTransfer(
    Block *block, ProgramPoint *point, Block *predecessor,
    const MemoryDependenceLattice &before, MemoryDependenceLattice *after) {
  DUMP_STATE_HELPER("block", block);

  assert(false && "block transfer not supported atm");
  return;
}

void MemoryDependenceAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const MemoryDependenceLattice &before, MemoryDependenceLattice *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()));

  assert(false && "call control flow transfer not supported atm");
}

void MemoryDependenceAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const MemoryDependenceLattice &before,
    MemoryDependenceLattice *after) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));

  assert(false && "region branch control flow transfer not supported atm");
}

void MemoryDependenceAnalysis::setToEntryState(
    MemoryDependenceLattice *lattice) {
  // At entry, everything is empty.
  // Still, mark it as changed so the forward analysis kicks off.
  propagateIfChanged(lattice, ChangeResult::Change);
}

#undef DUMP_STATE_HELPER
