//===- ForwardStoreToLoad.cpp - Store-to-load forwarding ------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Intra-block reaching-definition analysis on memref values (allocas and block
// args). Replaces loads whose value is already known from a prior store in the
// same block, and eliminates dead stores that are overwritten before being
// read.
//
// Handles both scalar memrefs (rank-0 or [1]) and indexed memrefs (rank-1 with
// static shape and constant indices), tracking reaching definitions per
// (memref, index) slot.
//
//   store %v1, %alloca[%c0]   // slot (alloca, 0)
//   load %alloca[%c0] -> %v1' // redundant, forwarded
//   compute(%v1') -> %v2
//   store %v2, %alloca[%c0]   // overwrites slot (alloca, 0)
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/MemRefUtils.h"
#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
#define GEN_PASS_DEF_FORWARDSTORETOLOAD
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {

/// A (memref, index) pair identifying a specific memory slot.
using MemRefSlot = std::pair<Value, int64_t>;

/// Returns the memref Value if it has a trackable type: rank-0, or rank-1
/// with static shape. Returns nullptr otherwise.
static Value getTrackableMemRef(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type)
    return nullptr;
  if (type.getRank() == 0)
    return memref;
  if (type.getRank() == 1 && type.hasStaticShape())
    return memref;
  return nullptr;
}

/// Get the slot index for a memref access. Returns 0 for rank-0 and [1]
/// memrefs, the constant index for indexed rank-1 memrefs, or std::nullopt
/// if the index is dynamic or out of range.
static std::optional<int64_t> getSlotIndex(MemRefType type,
                                           ValueRange indices) {
  if (type.getRank() == 0)
    return 0;
  if (type.getRank() == 1 && indices.size() == 1) {
    if (type.getShape()[0] == 1)
      return 0;
    if (auto cst = indices[0].getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t idx = cst.value();
      if (idx >= 0 && idx < type.getShape()[0])
        return idx;
    }
  }
  return std::nullopt;
}

/// Get the trackable slot for a store or load. Returns the (memref, index) pair
/// if the access is trackable, or std::nullopt otherwise.
static std::optional<MemRefSlot> getSlot(Value memref, ValueRange indices) {
  Value tracked = getTrackableMemRef(memref);
  if (!tracked)
    return std::nullopt;
  auto type = cast<MemRefType>(tracked.getType());
  auto idx = getSlotIndex(type, indices);
  if (!idx)
    return std::nullopt;
  return MemRefSlot{tracked, *idx};
}

/// Returns true if the memref is safe to optimize: all users are memref.load,
/// memref.store (with the memref as the memref operand), or scf.yield. If the
/// memref escapes (e.g., passed to a function call), we cannot safely forward.
static bool isNonEscaping(Value memref) {
  return llvm::all_of(memref.getUses(), [&](OpOperand &use) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<memref::StoreOp>(user))
      return store.getMemRef() == memref;
    if (auto load = dyn_cast<memref::LoadOp>(user))
      return load.getMemRef() == memref;
    if (isa<scf::YieldOp>(user))
      return true;
    return false;
  });
}

struct ForwardStoreToLoad
    : public mlir::aster::impl::ForwardStoreToLoadBase<ForwardStoreToLoad> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  void processBlock(Block &block);
};

void ForwardStoreToLoad::processBlock(Block &block) {
  // Cache safety checks per memref (avoid re-walking users).
  DenseMap<Value, bool> safetyCache;
  auto isSafe = [&](Value memref) -> bool {
    auto it = safetyCache.find(memref);
    if (it != safetyCache.end())
      return it->second;
    bool safe = isNonEscaping(memref);
    safetyCache[memref] = safe;
    return safe;
  };

  // reaching[slot] = last stored value (the reaching definition).
  DenseMap<MemRefSlot, Value> reaching;
  // unreadStores[slot] = the store op that wrote the reaching definition.
  // If this store is overwritten before being read, it is dead.
  DenseMap<MemRefSlot, memref::StoreOp> unreadStores;

  SmallVector<Operation *> toErase;

  for (Operation &op : block) {
    if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto slot = getSlot(store.getMemRef(), store.getIndices());
      if (!slot || !isSafe(slot->first))
        continue;

      // WAW: previous store to same slot overwritten before read -> dead.
      auto it = unreadStores.find(*slot);
      if (it != unreadStores.end())
        toErase.push_back(it->second);

      reaching[*slot] = store.getValueToStore();
      unreadStores[*slot] = store;
      continue;
    }

    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      auto slot = getSlot(load.getMemRef(), load.getIndices());
      if (!slot || !isSafe(slot->first))
        continue;

      // RAW: reaching def available -> forward stored value, erase load.
      auto it = reaching.find(*slot);
      if (it != reaching.end()) {
        load.getResult().replaceAllUsesWith(it->second);
        toErase.push_back(load);
      } else {
        // No reaching def yet. Any preceding store is live (not dead).
        assert(unreadStores.count(*slot) == 0 &&
               "unexpected pending unreadStore");
      }
      continue;
    }

    // For ops with regions (scf.for, scf.if, etc.), invalidate reaching defs
    // for any memref used inside the regions. Invalidate ALL slots for that
    // memref since nested code could access any index.
    if (op.getNumRegions() > 0) {
      DenseSet<Value> toInvalidate;
      for (Region &region : op.getRegions()) {
        region.walk([&](Operation *nested) {
          Value memref = nullptr;
          if (auto nestedLoad = dyn_cast<memref::LoadOp>(nested))
            memref = getTrackableMemRef(nestedLoad.getMemRef());
          else if (auto nestedStore = dyn_cast<memref::StoreOp>(nested))
            memref = getTrackableMemRef(nestedStore.getMemRef());
          if (memref)
            toInvalidate.insert(memref);
        });
      }
      // Erase all slots for invalidated memrefs.
      SmallVector<MemRefSlot> slotsToErase;
      for (auto &[slot, _] : reaching) {
        if (toInvalidate.contains(slot.first))
          slotsToErase.push_back(slot);
      }
      for (auto &slot : slotsToErase) {
        reaching.erase(slot);
        unreadStores.erase(slot);
      }
    }
  }

  for (Operation *op : llvm::reverse(toErase))
    op->erase();
}

void ForwardStoreToLoad::runOnOperation() {
  getOperation()->walk([&](Block *block) { processBlock(*block); });
  // After forwarding, some memref values (allocas, block args, op results)
  // may have no remaining loads. Eliminate dead stores to them, and dead
  // allocas, so canonicalize can remove unused iter_args.
  eliminateDeadMemrefStores(getOperation());
}

} // namespace
