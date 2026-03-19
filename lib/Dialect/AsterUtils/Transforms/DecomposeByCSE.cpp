//===- DecomposeByCSE.cpp - Common sub-expression decomposition -----------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "decompose-by-cse"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_DECOMPOSEBYCSE
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
struct DecomposeByCSE
    : public aster_utils::impl::DecomposeByCSEBase<DecomposeByCSE> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Common sub-expression decomposition
//===----------------------------------------------------------------------===//

/// Implements the common sub-expression decomposition for a single basic block
/// and a single op kind (AddiOp or MuliOp).
template <typename OpTy>
struct DecomposeCSEImpl {
  DecomposeCSEImpl(IRRewriter &rewriter, Block *block)
      : rewriter(rewriter), block(block) {}

  /// Run the common sub-expression decomposition.
  void run();

private:
  /// Collect all ops of the given kind in the block into `ops` and initialize
  /// internal structures. Populate the operands and operandToExpr map.
  void collectOps();

  /// Run the fixpoint algorithm to find the most profitable maximal common
  /// sub-expression.
  void runCSEFixpoint();

  /// Materialize the common sub-expression ops into the block.
  void materialize();

  /// Get or add a sub-expression ID for the pair of operands.
  int32_t getOrAddSubExpr(int32_t a, int32_t b);

  /// Expand a sub-expression ID to a flat list of Values.
  void expandSubExpression(Location loc, int32_t id,
                           SmallVectorImpl<Value> &result,
                           const DenseSet<int32_t> &usedExprs,
                           DenseMap<int32_t, Value> &barrierPoints);

  IRRewriter &rewriter;
  /// The block being processed.
  Block *block;
  /// The ops to process.
  SmallVector<OpTy> ops;
  /// The unique values in the block corresponding to the operands of the ops.
  SetVector<Value> uniqueValues;
  /// The operands of the ops in the block, represented as sorted lists of
  /// unique value indices.
  SmallVector<SmallVector<int32_t>> opOperands;
  /// A map from values to their unique index in `uniqueValues`.
  DenseMap<Value, int32_t> valuePosMap;
  /// The sub-expressions found in the block.
  SetVector<std::pair<int32_t, int32_t>> subExprs;
  DenseMap<std::pair<int32_t, int32_t>, int32_t> subExprPosMap;
};

template <typename OpTy>
void DecomposeCSEImpl<OpTy>::run() {
  collectOps();
  if (ops.size() < 2)
    return;
  runCSEFixpoint();
  materialize();
}

template <typename OpTy>
void DecomposeCSEImpl<OpTy>::collectOps() {
  for (OpTy op : block->getOps<OpTy>()) {
    ops.push_back(op);

    // Collect the operands and their unique indices.
    SmallVector<int32_t> ids;
    for (Value v : op.getInputs()) {
      auto [it, inserted] =
          valuePosMap.try_emplace(v, static_cast<int32_t>(uniqueValues.size()));
      if (inserted)
        uniqueValues.insert(v);
      ids.push_back(it->second);
    }
    llvm::sort(ids);
    opOperands.push_back(std::move(ids));
  }

  // Initialize the sub-expression map with identity pairs {i,i} for each
  // operand.
  for (int32_t i = 0; i < static_cast<int32_t>(uniqueValues.size()); ++i)
    getOrAddSubExpr(i, i);
}

template <typename OpTy>
int32_t DecomposeCSEImpl<OpTy>::getOrAddSubExpr(int32_t a, int32_t b) {
  if (a > b)
    std::swap(a, b);
  auto [it, inserted] =
      subExprPosMap.try_emplace({a, b}, static_cast<int32_t>(subExprs.size()));
  if (inserted)
    subExprs.insert({a, b});
  return it->second;
}

template <typename OpTy>
void DecomposeCSEImpl<OpTy>::runCSEFixpoint() {
  DenseMap<std::pair<int32_t, int32_t>, int32_t> pairFreq;
  while (true) {
    pairFreq.clear();

    // Count the frequency of each pair of IDs.
    for (const SmallVector<int32_t> &ids : opOperands) {
      for (int32_t j = 0; j < static_cast<int32_t>(ids.size()); ++j) {
        for (int32_t k = j + 1; k < static_cast<int32_t>(ids.size()); ++k)
          ++pairFreq[{ids[j], ids[k]}];
      }
    }

    // Find the most frequent pair of IDs with frequency >= 2.
    std::pair<int32_t, int32_t> best = {-1, -1};
    int32_t bestFreq = 1;
    for (auto &[pair, freq] : pairFreq) {
      if (freq > bestFreq) {
        bestFreq = freq;
        best = pair;
      }
    }

    // If no pair of IDs has frequency >= 2, we are done.
    if (best.first == -1) {
      LDBG() << "No pair of IDs has frequency >= 2, done";
      break;
    }

    // Add the new sub-expression to the map and get its ID.
    int32_t newId = getOrAddSubExpr(best.first, best.second);
    LDBG() << "CSE fixpoint: pair=(" << best.first << "," << best.second
           << ") freq=" << bestFreq << " newId=" << newId;

    // Replace both component IDs with newId in every matching op.
    // Since newId > all existing IDs, appending it keeps the list sorted.
    for (SmallVector<int32_t> &ids : opOperands) {
      auto itA = llvm::find(ids, best.first);
      auto itB = llvm::find(ids, best.second);
      // If none of the IDs are found, this is not a sub-expression of this op.
      if (itA == ids.end() || itB == ids.end())
        continue;
      // Remove the old IDs and add the new one.
      ids.erase(itA);
      ids.erase(llvm::find(ids, best.second));
      ids.push_back(newId);
    }
  }
}

template <typename OpTy>
void DecomposeCSEImpl<OpTy>::expandSubExpression(
    Location loc, int32_t id, SmallVectorImpl<Value> &result,
    const DenseSet<int32_t> &usedExprs,
    DenseMap<int32_t, Value> &barrierPoints) {
  int32_t a = subExprs[id].first;
  int32_t b = subExprs[id].second;
  LDBG() << "Expanding sub-expression: " << id << " = (" << a << "," << b
         << ")";

  // If the sub-expression is an identity pair, add the unique value to the
  // result.
  if (a == b) {
    result.push_back(uniqueValues[a]);
    return;
  }

  // Helper function to get or create a barrier for a sub-expression.
  auto getOrCreateBarrier = [&](int32_t id,
                                ArrayRef<int32_t> collectList) -> Value {
    Value barrier = barrierPoints.lookup(id);
    // If the barrier is not found, recursively expand the child and create a
    // new barrier.
    if (!barrier) {
      // Collect the operands for the new barrier using the IDs in the collect
      // list.
      SmallVector<Value> inputs;
      for (int32_t child : collectList)
        expandSubExpression(loc, child, inputs, usedExprs, barrierPoints);

      // Create the new barrier op.
      Value op = OpTy::create(rewriter, loc, rewriter.getIndexType(), inputs);
      barrier = PassthroughOp::create(
          rewriter, loc, op, rewriter.getStringAttr("__decompose_ops__"));
      barrierPoints[id] = barrier;
    }
    return barrier;
  };

  // If the sub-expression is live, add the barrier value to the
  // result.
  if (usedExprs.contains(id)) {
    result.push_back(getOrCreateBarrier(id, {a, b}));
    return;
  }

  // Recursively expand the sub-expressions.
  for (int32_t child : {a, b})
    expandSubExpression(loc, child, result, usedExprs, barrierPoints);
}

template <typename OpTy>
void DecomposeCSEImpl<OpTy>::materialize() {
  // Collect all the IDs that appear in at least one op.
  DenseSet<int32_t> usedExprs;
  for (const SmallVector<int32_t> &ids : opOperands) {
    for (int32_t id : ids) {
      if (subExprs[id].first != subExprs[id].second)
        usedExprs.insert(id);
    }
  }

  // If no IDs are live, we are done.
  if (usedExprs.empty())
    return;

  Type indexType = rewriter.getIndexType();
  OpBuilder::InsertionGuard guard(rewriter);
  DenseMap<int32_t, Value> barrierPoints;
  SmallVector<Value> operands, replacements;

  // Create the replacements for the original ops.
  for (auto [op, ids] : llvm::zip(ops, opOperands)) {
    rewriter.setInsertionPoint(op);
    operands.clear();
    for (int32_t id : ids)
      expandSubExpression(op.getLoc(), id, operands, usedExprs, barrierPoints);

    LDBG() << "Processing op: " << op;
    LDBG() << "Operands: " << llvm::interleaved_array(operands);

    if (operands.size() == 1) {
      replacements.push_back(operands[0]);
      continue;
    }
    replacements.push_back(
        OpTy::create(rewriter, op.getLoc(), indexType, operands));
  }

  // Replace the original ops with the replacements in reverse order to ensure
  // the replacements are correct.
  for (auto [op, replacement] : llvm::reverse(llvm::zip(ops, replacements)))
    rewriter.replaceOp(op, replacement);
}

//===----------------------------------------------------------------------===//
// DecomposeByCSE pass
//===----------------------------------------------------------------------===//

void DecomposeByCSE::runOnOperation() {
  IRRewriter rewriter(&getContext());
  getOperation()->walk([&](Block *block) {
    DecomposeCSEImpl<AddiOp>(rewriter, block).run();
    DecomposeCSEImpl<MuliOp>(rewriter, block).run();
  });
}
