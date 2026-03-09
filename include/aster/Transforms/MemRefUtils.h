//===- MemRefUtils.h - Shared memref analysis utilities -------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reusable helpers for memref analysis and transformation passes:
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_MEMREFUTILS_H
#define ASTER_TRANSFORMS_MEMREFUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {

/// Returns true if the memref type is scalar: rank-0 or rank-1 with shape [1].
inline bool isScalarMemRef(MemRefType type) {
  if (type.getRank() == 0)
    return true;
  if (type.getRank() == 1 && type.getShape()[0] == 1)
    return true;
  return false;
}

/// Forward stores from storeTarget[i] to loads from loadSource[i], where both
/// refer to the same underlying memory (e.g., a static alloca and its dynamic
/// cast, or the same value for self-forwarding). Both must be in the same basic
/// block with stores dominating loads. Erases forwarded loads and dead stores.
/// Returns success if forwarding succeeded (or there was nothing to forward).
LogicalResult forwardConstantIndexStores(IRRewriter &rewriter,
                                         Value storeTarget, Value loadSource,
                                         int64_t numElements);

/// Erase all stores to a memref value that has no load users. The memref may
/// have other non-escaping users (scf.yield, scf.for, memref.cast). Returns
/// success if any stores were erased.
/// Warning: this is optimistic, yse when the caller knows the alias context
/// (e.g., after forwarding made all loads dead for a specific iter_arg).
LogicalResult eraseDeadMemrefStores(IRRewriter &rewriter, Value memref);

/// Walk all memref.store ops under `root` and eliminate stores to memref values
/// that have only store users (no loads, yields, or other escaping uses).
/// Also erases dead allocas whose only users were the eliminated stores.
/// More conservative than eraseDeadMemrefStores: a yielded or for-passed memref
/// is considered live because it may alias a value with loads elsewhere.
void eliminateDeadMemrefStores(Operation *root);

} // namespace mlir::aster

#endif // ASTER_TRANSFORMS_MEMREFUTILS_H
