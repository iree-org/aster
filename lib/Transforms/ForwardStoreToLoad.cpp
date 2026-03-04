//===- ForwardStoreToLoad.cpp - Scalar store-to-load forwarding -----------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Intra-block reaching-definition analysis on scalar memref.alloca ops.
// Replaces loads whose value is already known from a prior store in the same
// block, and eliminates dead stores that are overwritten before being read.
//
//   store %v1, %alloca[]      // kt=0 MFMA result
//   load %alloca[] -> %v1'    // kt=1 input (redundant)
//   mfma(%v1') -> %v2
//   store %v2, %alloca[]      // kt=1 result
//
// This pass forwards %v1 to the load, then the dead first store is eliminated.
// The result is 1 load + 1 store per alloca, which PromoteLoopCarriedMemrefs
// can then convert to iter_args.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// Returns true if the memref type is scalar: rank-0 or rank-1 with shape [1].
static bool isScalarMemRef(MemRefType type) {
  if (type.getRank() == 0)
    return true;
  if (type.getRank() == 1 && type.getShape()[0] == 1)
    return true;
  return false;
}

/// Returns the alloca Value if the given memref value is a scalar
/// memref.alloca, otherwise returns nullptr.
static Value getScalarAlloca(Value memref) {
  auto allocaOp = memref.getDefiningOp<memref::AllocaOp>();
  if (!allocaOp)
    return nullptr;
  if (!isScalarMemRef(allocaOp.getType()))
    return nullptr;
  return allocaOp.getResult();
}

/// Returns true if the alloca is safe to optimize: all users are memref.load
/// or memref.store ops with the alloca as the memref operand. If the alloca
/// escapes (e.g., passed to a function call), we cannot safely forward or
/// eliminate stores.
static bool isNonEscapingAlloca(Value alloca) {
  return llvm::all_of(alloca.getUses(), [&](OpOperand &use) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<memref::StoreOp>(user))
      return store.getMemRef() == alloca;
    if (auto load = dyn_cast<memref::LoadOp>(user))
      return load.getMemRef() == alloca;
    return false;
  });
}

struct ForwardStoreToLoad
    : public mlir::aster::impl::ForwardStoreToLoadBase<ForwardStoreToLoad> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Process a single block: forward stores to loads and eliminate dead stores.
  /// Returns true if any changes were made.
  void processBlock(Block &block);
};

void ForwardStoreToLoad::processBlock(Block &block) {
  // Cache safety checks per alloca (avoid re-walking users).
  DenseMap<Value, bool> safetyCache;
  auto isSafe = [&](Value alloca) -> bool {
    auto it = safetyCache.find(alloca);
    if (it != safetyCache.end())
      return it->second;
    bool safe = isNonEscapingAlloca(alloca);
    safetyCache[alloca] = safe;
    return safe;
  };

  // reaching[alloca] = last stored value (the reaching definition).
  DenseMap<Value, Value> reaching;
  // unreadStores[alloca] = the store op that wrote the reaching definition.
  // If this store is overwritten before being read from memory, it is dead.
  DenseMap<Value, memref::StoreOp> unreadStores;

  SmallVector<Operation *> toErase;

  // Walk ops in order within the block and nothing else.
  for (Operation &op : block) {
    if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      Value alloca = getScalarAlloca(store.getMemRef());
      if (!alloca || !isSafe(alloca))
        continue;

      // WAW: previous store is overwritten before being read -> dead store.
      //   store %v1, %alloca[]   <-- dead, erased
      //   store %v2, %alloca[]   <-- overwrites without intervening load
      auto it = unreadStores.find(alloca);
      if (it != unreadStores.end())
        toErase.push_back(it->second);

      reaching[alloca] = store.getValueToStore();
      unreadStores[alloca] = store;
      continue;
    }

    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      Value alloca = getScalarAlloca(load.getMemRef());
      if (!alloca || !isSafe(alloca))
        continue;

      // RAW: reaching def available -> forward stored value, erase load.
      //   store %v1, %alloca[]
      //   %v2 = load %alloca[]   <-- replaced with %v1, erased
      auto it = reaching.find(alloca);
      if (it != reaching.end()) {
        load.getResult().replaceAllUsesWith(it->second);
        toErase.push_back(load);
        // Do NOT clear unreadStores: the load was forwarded and erased, so the
        // preceding store's write to memory is unobserved. The next store to
        // this alloca can safely mark it dead.
      } else {
        // No reaching def (i.e. no store at this place yet): the value either
        // comes from a prior block or this is a first access (unit. data).
        // This load is a real memory read, so any preceding store is live and
        // is not subject to WAW considerations. Ensure it.
        assert(unreadStores.count(alloca) == 0 &&
               "unexpected pending unreadStore to alloca");
      }
      continue;
    }

    // For ops with regions (scf.for, scf.if, etc.), the nested region may
    // load from or store to tracked allocas. We must invalidate reaching
    // defs for any alloca used inside the regions: we do not yet model
    // control-flow for this.
    if (op.getNumRegions() > 0) {
      SmallVector<Value> toInvalidate;
      for (Region &region : op.getRegions()) {
        region.walk([&](Operation *nested) {
          if (auto nestedLoad = dyn_cast<memref::LoadOp>(nested)) {
            Value alloca = getScalarAlloca(nestedLoad.getMemRef());
            if (alloca && reaching.count(alloca))
              toInvalidate.push_back(alloca);
          } else if (auto nestedStore = dyn_cast<memref::StoreOp>(nested)) {
            Value alloca = getScalarAlloca(nestedStore.getMemRef());
            if (alloca && reaching.count(alloca))
              toInvalidate.push_back(alloca);
          }
        });
      }
      for (Value alloca : toInvalidate) {
        reaching.erase(alloca);
        unreadStores.erase(alloca);
      }
    }
  }

  for (Operation *op : llvm::reverse(toErase))
    op->erase();
}

void ForwardStoreToLoad::runOnOperation() {
  getOperation()->walk([&](Block *block) { processBlock(*block); });
}

} // namespace
