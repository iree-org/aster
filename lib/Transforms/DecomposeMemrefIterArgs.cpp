//===- DecomposeMemrefIterArgs.cpp - Decompose memref iter_args -----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposes memref<NxT> iter_args into N scalar T iter_args via
// replaceWithAdditionalYields, then lets canonicalize remove the dead memref
// iter_args.
//
// Pattern: yield = fresh alloca with stores at [0, N), loads from block arg.
// Also handles self-forwarding (stores and loads on the same block arg).
//
// Run aster-simplify-alloca-iter-args first to fold casts and dedup.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/MemRefUtils.h"
#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-decompose-memref-iter-args"

namespace mlir::aster {
#define GEN_PASS_DEF_DECOMPOSEMEMREFITERARGS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Collect values stored to `target` at constant indices [0, numElements).
/// Returns empty vector if any index is missing, duplicated, or non-constant.
static SmallVector<Value> collectStoredValues(Value target,
                                              int64_t numElements) {
  SmallVector<Value> values(numElements, nullptr);
  for (Operation *user : target.getUsers()) {
    auto store = dyn_cast<memref::StoreOp>(user);
    if (!store || store.getMemRef() != target)
      continue;
    if (store.getIndices().size() != 1)
      return {};
    auto idx = getConstantIntValue(getAsOpFoldResult(store.getIndices()[0]));
    if (!idx || *idx < 0 || *idx >= numElements)
      return {};
    if (values[*idx])
      return {};
    values[*idx] = store.getValueToStore();
  }
  for (Value v : values)
    if (!v)
      return {};
  return values;
}

/// Replace all memref.load ops on `memref` with the corresponding value from
/// `replacements` (indexed by constant load index). Erases the replaced loads.
static void replaceConstantIndexLoads(IRRewriter &rewriter, Value memref,
                                      ValueRange replacements) {
  SmallVector<memref::LoadOp> loads;
  for (OpOperand &use : memref.getUses())
    if (auto load = dyn_cast<memref::LoadOp>(use.getOwner()))
      if (load.getMemRef() == memref)
        loads.push_back(load);

  for (auto load : loads) {
    auto idx = getConstantIntValue(getAsOpFoldResult(load.getIndices()[0]));
    assert(idx && "expected constant index load");
    rewriter.replaceAllUsesWith(load.getResult(), replacements[*idx]);
    rewriter.eraseOp(load);
  }
}

/// Return the static 1-D memref type of a value, or nullptr.
static MemRefType getStatic1DMemRefType(Value v) {
  auto type = dyn_cast<MemRefType>(v.getType());
  if (type && type.getRank() == 1 && type.hasStaticShape())
    return type;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

struct DecomposableSlot {
  int64_t argIdx;
  SmallVector<Value> initValues;
  SmallVector<Value> yieldValues;
};

static SmallVector<DecomposableSlot> findDecomposableSlots(scf::ForOp forOp) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<DecomposableSlot> slots;

  for (int64_t i = 0, n = forOp.getNumRegionIterArgs(); i < n; ++i) {
    auto memrefType = getStatic1DMemRefType(forOp.getInitArgs()[i]);
    if (!memrefType)
      continue;
    int64_t numElems = memrefType.getShape()[0];

    Value yieldVal = yieldOp.getOperand(i);
    if (!yieldVal.getDefiningOp<memref::AllocaOp>())
      continue;
    auto ys = collectStoredValues(yieldVal, numElems);
    if (ys.empty())
      continue;
    auto is = collectStoredValues(forOp.getInitArgs()[i], numElems);
    if (is.empty()) {
      LDBG() << "  SKIP iter_arg #" << i << ": init lacks complete stores";
      continue;
    }

    LDBG() << "  Decomposable iter_arg #" << i << " (" << numElems
           << " elements)";
    slots.push_back({i, std::move(is), std::move(ys)});
  }
  return slots;
}

//===----------------------------------------------------------------------===//
// Decomposition via replaceWithAdditionalYields
//===----------------------------------------------------------------------===//

static bool decomposeMemrefIterArgs(scf::ForOp forOp) {
  SmallVector<DecomposableSlot> slots = findDecomposableSlots(forOp);
  if (slots.empty())
    return false;

  IRRewriter rewriter(forOp.getContext());
  int64_t numOldArgs = forOp.getNumRegionIterArgs();

  // Collect scalar init values for all decomposable slots.
  SmallVector<Value> newInits;
  for (auto &slot : slots)
    newInits.append(slot.initValues.begin(), slot.initValues.end());

  // Add scalar iter_args alongside existing memref ones.
  // The callback returns yield values for the new scalar iter_args (the values
  // stored to the yield alloca). Called before body is moved.
  auto yieldFn = [&](OpBuilder &, Location,
                     ArrayRef<BlockArgument>) -> SmallVector<Value> {
    SmallVector<Value> yields;
    for (auto &slot : slots)
      yields.append(slot.yieldValues.begin(), slot.yieldValues.end());
    return yields;
  };

  auto result = forOp.replaceWithAdditionalYields(
      rewriter, newInits, /*replaceInitOperandUsesInLoop=*/false, yieldFn);
  assert(succeeded(result));
  auto newForOp = cast<scf::ForOp>(result->getOperation());

  // Replace memref loads with new scalar block args (in-loop) and scalar
  // results (post-loop). The dead memref iter_args are removed by canonicalize.
  int64_t scalarStart = numOldArgs;
  for (auto &slot : slots) {
    int64_t n = slot.initValues.size();
    replaceConstantIndexLoads(
        rewriter, newForOp.getRegionIterArgs()[slot.argIdx],
        newForOp.getRegionIterArgs().slice(scalarStart, n));
    replaceConstantIndexLoads(rewriter, newForOp.getResult(slot.argIdx),
                              newForOp.getResults().slice(scalarStart, n));
    (void)eraseDeadMemrefStores(rewriter, newForOp.getResult(slot.argIdx));
    scalarStart += n;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Self-forwarding: stores and loads on the same block arg
//===----------------------------------------------------------------------===//

static void forwardSameBlockArgStores(scf::ForOp forOp) {
  IRRewriter rewriter(forOp.getContext());

  for (int64_t i = 0, n = forOp.getNumRegionIterArgs(); i < n; ++i) {
    auto memrefType = getStatic1DMemRefType(forOp.getInitArgs()[i]);
    if (!memrefType)
      continue;

    int64_t numElements = memrefType.getShape()[0];
    BlockArgument ba = forOp.getRegionIterArgs()[i];

    if (failed(forwardConstantIndexStores(rewriter, ba, ba, numElements)))
      continue;
    LDBG() << "  Self-forwarded iter_arg #" << i;

    (void)eraseDeadMemrefStores(rewriter, ba);
    Value init = forOp.getInitArgs()[i];
    Value result = forOp.getResult(i);
    (void)forwardConstantIndexStores(rewriter, result, result, numElements);
    (void)forwardConstantIndexStores(rewriter, init, result, numElements);
    (void)eraseDeadMemrefStores(rewriter, result);
    (void)eraseDeadMemrefStores(rewriter, init);
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DecomposeMemrefIterArgs
    : public mlir::aster::impl::DecomposeMemrefIterArgsBase<
          DecomposeMemrefIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void DecomposeMemrefIterArgs::runOnOperation() {
  Operation *rootOp = getOperation();

  {
    SmallVector<scf::ForOp> forOps;
    rootOp->walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps)
      decomposeMemrefIterArgs(forOp);
  }

  {
    SmallVector<scf::ForOp> forOps;
    rootOp->walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps)
      forwardSameBlockArgStores(forOp);
  }

  MLIRContext *ctx = rootOp->getContext();
  RewritePatternSet patterns(ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  memref::AllocaOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantOp::getCanonicalizationPatterns(patterns, ctx);
  memref::StoreOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(rootOp, std::move(patterns));
}
