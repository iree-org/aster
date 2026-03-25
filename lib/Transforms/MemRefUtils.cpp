//===- MemRefUtils.cpp - Shared memref analysis utilities -----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/MemRefUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult mlir::aster::forwardConstantIndexStores(IRRewriter &rewriter,
                                                      Value storeTarget,
                                                      Value loadSource,
                                                      int64_t numElements) {
  // Collect stores to storeTarget at constant indices.
  SmallVector<memref::StoreOp> stores(numElements, nullptr);
  for (Operation *user : storeTarget.getUsers()) {
    auto storeOp = dyn_cast<memref::StoreOp>(user);
    if (!storeOp)
      continue; // Allow non-store users (e.g., scf.yield)
    if (storeOp.getIndices().size() != 1)
      return failure();
    auto idx = getConstantIntValue(getAsOpFoldResult(storeOp.getIndices()[0]));
    if (!idx || *idx < 0 || *idx >= numElements)
      return failure();
    if (stores[*idx]) // Multiple stores to the same index - bail out
      return failure();
    stores[*idx] = storeOp;
  }

  // Collect loads from loadSource.
  SmallVector<memref::LoadOp> loads;
  for (Operation *user : loadSource.getUsers()) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
      loads.push_back(loadOp);
      continue;
    }
    // Allow scf.yield (the memref may be yielded for rotation).
    if (isa<scf::YieldOp>(user))
      continue;
    // Allow stores to loadSource (self-forwarding: storeTarget == loadSource).
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      if (storeOp.getMemRef() == loadSource)
        continue;
    }
    // Allow scf.for (memref passed as init arg).
    if (isa<scf::ForOp>(user))
      continue;
    // Unexpected user.
    return failure();
  }

  // No loads -> nothing to forward, still success.
  if (loads.empty())
    return success();

  // Verify and forward each load.
  for (auto loadOp : loads) {
    if (loadOp.getIndices().size() != 1)
      return failure();
    auto idx = getConstantIntValue(getAsOpFoldResult(loadOp.getIndices()[0]));
    if (!idx || *idx < 0 || *idx >= numElements)
      return failure();
    if (!stores[*idx])
      return failure(); // No store at this index

    // Verify store dominates load (same block, program order).
    if (stores[*idx]->getBlock() != loadOp->getBlock() ||
        !stores[*idx]->isBeforeInBlock(loadOp))
      return failure();

    rewriter.replaceAllUsesWith(loadOp.getResult(),
                                stores[*idx].getValueToStore());
  }

  // Erase forwarded loads.
  for (auto loadOp : loads)
    rewriter.eraseOp(loadOp);

  // Erase dead stores (the alloca has no readers after forwarding).
  for (auto storeOp : stores)
    if (storeOp)
      rewriter.eraseOp(storeOp);

  return success();
}

LogicalResult mlir::aster::eraseDeadMemrefStores(IRRewriter &rewriter,
                                                 Value memref) {
  SmallVector<memref::StoreOp> stores;
  for (OpOperand &use : memref.getUses()) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemRef() == memref) {
        stores.push_back(store);
        continue;
      }
      // Used as value-to-store, not as target -- not a dead store pattern.
      return failure();
    }
    if (isa<memref::LoadOp>(user))
      return failure(); // Has loads -- stores are live.
    // Allow yield, for, cast, etc. -- these don't read the stored data.
  }
  if (stores.empty())
    return failure();
  for (auto store : stores)
    rewriter.eraseOp(store);
  return success();
}
