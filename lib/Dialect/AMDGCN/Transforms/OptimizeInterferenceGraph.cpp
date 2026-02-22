//===- OptimizeInterferenceGraph.cpp - Optimize interference graph --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/InstImpl.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#include <algorithm>

#define DEBUG_TYPE "amdgcn-interference-opt"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
using NodeID = RegisterInterferenceGraph::NodeID;

/// Move operation descriptor.
struct MovDesc {
  MovDesc(Operation *moveOp, ValueRange srcAllocas, ValueRange targetAllocas,
          int32_t priority = 1)
      : moveOp(moveOp), srcAllocas(srcAllocas), targetAllocas(targetAllocas),
        priority(priority) {}
  Operation *moveOp;
  ValueRange srcAllocas;
  ValueRange targetAllocas;
  int32_t priority;
};

struct OptimizeGraphImpl {
  OptimizeGraphImpl(const RegisterInterferenceGraph &graph,
                    DataFlowSolver &solver)
      : graph(graph), solver(solver) {}

  /// Build equivalence classes for the interference graph.
  std::optional<IntEquivalenceClasses> run(Operation *root);

  /// Handle a move operation.
  LogicalResult handleMov(Operation *op, std::pair<Value, Value> moveInfo);

  /// Optimize the graph and populate the equivalence classes.
  void optimizeGraph(IntEquivalenceClasses &eqClasses);

  const RegisterInterferenceGraph &graph;
  DataFlowSolver &solver;
  SmallVector<MovDesc> movOps;
};
} // namespace

static std::optional<std::pair<Value, Value>> getMoveInfo(Operation *op) {
  if (auto copyOp = dyn_cast<lsir::CopyOp>(op))
    return std::make_pair(copyOp.getSource(), copyOp.getTarget());
  if (auto vop1 = dyn_cast<inst::VOP1Op>(op))
    return std::make_pair(vop1.getSrc0(), vop1.getVdst());
  if (auto sop1 = dyn_cast<inst::SOP1Op>(op))
    return std::make_pair(sop1.getSrc0(), sop1.getSdst());
  return std::nullopt;
}

LogicalResult OptimizeGraphImpl::handleMov(Operation *op,
                                           std::pair<Value, Value> moveInfo) {
  FailureOr<ValueRange> srcAlloc = getAllocasOrFailure(moveInfo.first);
  if (failed(srcAlloc))
    return failure();
  FailureOr<ValueRange> tgtAlloc = getAllocasOrFailure(moveInfo.second);
  if (failed(tgtAlloc))
    return failure();

  int32_t priority = 1;

  // If the source allocation has load reaching definitions, set the priority to
  // 0. Moves with lower priority are coalesced first.
  const auto *reachingDefs = solver.lookupState<ReachingDefinitionsState>(
      solver.getProgramPointBefore(op));
  if (reachingDefs && llvm::any_of(*srcAlloc, [&](Value value) {
        auto range = reachingDefs->getRange(value);
        return range.begin() != range.end();
      })) {
    priority = 0;
  }

  movOps.push_back({op, *srcAlloc, *tgtAlloc, priority});
  return success();
}

/// Check if the given classes have an edge between them.
static bool hasEdge(const Graph &graph, ArrayRef<int32_t> lhs,
                    ArrayRef<int32_t> rhs, IntEquivalenceClasses &eqClasses) {
  ArrayRef<int32_t> nodes = lhs.size() < rhs.size() ? lhs : rhs;
  int32_t leaderOther = lhs.size() < rhs.size() ? rhs[0] : lhs[0];
  for (int32_t node : nodes) {
    for (auto [src, tgt] : graph.edges(node)) {
      if (eqClasses.getLeader(tgt) == leaderOther)
        return true;
    }
  }
  return false;
}

/// Check if the given classes can be coalesced.
static bool canCoalesce(const RegisterInterferenceGraph &graph,
                        IntEquivalenceClasses &eqClasses, NodeID src,
                        NodeID tgt, int32_t size) {
  LDBG() << "-- Checking if can coalesce: " << src << " and " << tgt
         << " with size " << size;
  int32_t tmp0, tmp1;
  for (auto [srcId, tgtId] :
       llvm::zip_equal(llvm::seq<NodeID>(src, src + size),
                       llvm::seq<NodeID>(tgt, tgt + size))) {
    // If the nodes are the same continue.
    if (srcId == tgtId)
      continue;

    // Get the members of the classes.
    ArrayRef<int32_t> srcMembers = eqClasses.getMembers(srcId, tmp0);
    ArrayRef<int32_t> tgtMembers = eqClasses.getMembers(tgtId, tmp1);

    LDBG() << "--- Source members: " << llvm::interleaved_array(srcMembers);
    LDBG() << "--- Target members: " << llvm::interleaved_array(tgtMembers);

    // Continue if the classes are the same.
    if (srcMembers[0] == tgtMembers[0])
      continue;

    // Check if the classes have an edge between them.
    if (hasEdge(graph, srcMembers, tgtMembers, eqClasses)) {
      LDBG() << "--- Has edge between classes, cannot coalesce";
      return false;
    }
  }
  LDBG() << "--- Can coalesce";
  return true;
}

/// Get the bounds of the ranges to call canCoalesce on. This method returns
/// failure if the ranges are incompatible.
/// NOTE: We always try to merge the smallest range into the biggest
/// one.Otherwise, we would need to create a new range containing both ranges
/// and destroy the current ranges, which also implies a re-numbering of the
/// nodes in the graph.
/// TODO: Allow merging with re-numbering.
static FailureOr<std::tuple<NodeID, NodeID, int32_t>>
getRangeBounds(NodeID lhsBegin, int32_t lhsOff, int32_t lhsSize,
               int32_t lhsAlignment, NodeID rhsBegin, int32_t rhsOff,
               int32_t rhsSize, int32_t rhsAlignment) {
  /// Swap so that the lhs is the larger range.
  if (lhsSize < rhsSize)
    return getRangeBounds(rhsBegin, rhsOff, rhsSize, rhsAlignment, lhsBegin,
                          lhsOff, lhsSize, lhsAlignment);

  // Get the start of the rhs in the lhs according to the copy offset.
  int32_t rhsInLhsStart = lhsOff - rhsOff;

  // Bail if we cannot fit the elemnents before the rhs offset in the lhs range.
  if (rhsInLhsStart < 0)
    return failure();

  // Bail if we cannot fit the elements after the rhs offset in the lhs range.
  if (rhsInLhsStart + rhsSize > lhsSize)
    return failure();

  // Bail if the alignment is not compatible.
  if (rhsInLhsStart % rhsAlignment != 0)
    return failure();

  return std::make_tuple(static_cast<NodeID>(rhsInLhsStart + lhsBegin),
                         rhsBegin, rhsSize);
}

/// Optimize the graph by coalescing the registers involved in the moves.
/// NOTE: Given the prescene of register ranges, is not enough to only check the
/// registers involved in the moves, we need to check that one of the entire
/// ranges can be coalesced into one of them.
void OptimizeGraphImpl::optimizeGraph(IntEquivalenceClasses &eqClasses) {
  for (const MovDesc &mov : movOps) {
    LDBG() << "Processing move: " << *mov.moveOp;

    // NOTE: This is safe because the graph provides the guruantee that ranges
    // are consecutive.
    NodeID srcId = graph.getNodeId(mov.srcAllocas[0]);
    NodeID tgtId = graph.getNodeId(mov.targetAllocas[0]);
    LDBG() << "- Source ID: " << srcId << ", Target ID: " << tgtId;

    // If the source and target are the same, continue.
    if (srcId == tgtId)
      continue;

    auto [srcRangeId, srcRange] = graph.getRangeInfo(srcId);
    auto [tgtRangeId, tgtRange] = graph.getRangeInfo(tgtId);
    LDBG() << "- Source range ID: " << srcRangeId
           << ", Target range ID: " << tgtRangeId;

    // If the source and target are in the same range, continue. We cannot
    // coalesce members of the same range.
    if (srcRangeId == tgtRangeId)
      continue;

    // Get the start of the copy offsets.
    int32_t srcOffset = srcId - srcRangeId;
    int32_t tgtOffset = tgtId - tgtRangeId;
    LDBG() << "- Source offset: " << srcOffset
           << ", Target offset: " << tgtOffset;

    // Get the size of the ranges.
    int32_t srcRangeSize = srcRange ? srcRange->allocations.size() : 1;
    int32_t tgtRangeSize = tgtRange ? tgtRange->allocations.size() : 1;
    LDBG() << "- Source range size: " << srcRangeSize
           << ", Target range size: " << tgtRangeSize;

    // Get the alignment of the ranges.
    int32_t srcRangeAlignment = srcRange ? srcRange->alignment : 1;
    int32_t tgtRangeAlignment = tgtRange ? tgtRange->alignment : 1;
    LDBG() << "- Source range alignment: " << srcRangeAlignment
           << ", Target range alignment: " << tgtRangeAlignment;

    // Get the bounds for the range coalescing, bail if the ranges are
    // incompatible.
    FailureOr<std::tuple<NodeID, NodeID, int32_t>> bounds =
        getRangeBounds(srcRangeId, srcOffset, srcRangeSize, srcRangeAlignment,
                       tgtRangeId, tgtOffset, tgtRangeSize, tgtRangeAlignment);
    if (failed(bounds)) {
      LDBG() << "--- Ranges are incompatible";
      continue;
    }

    auto [lhsStart, rhsStart, size] = *bounds;
    LDBG() << "- Left start: " << lhsStart << ", Right start: " << rhsStart
           << ", Size: " << size;

    // Continue if the ranges cannot be coalesced.
    if (!canCoalesce(graph, eqClasses, lhsStart, rhsStart, size))
      continue;

    // Coalesce the ranges.
    for (auto [srcId, tgtId] :
         llvm::zip_equal(llvm::seq<NodeID>(lhsStart, lhsStart + size),
                         llvm::seq<NodeID>(rhsStart, rhsStart + size))) {
      eqClasses.join(srcId, tgtId);
      LDBG() << "--- Joined: " << srcId << " and " << tgtId;
    }
  }
}

std::optional<IntEquivalenceClasses> OptimizeGraphImpl::run(Operation *root) {
  WalkResult result = root->walk([&](Operation *op) -> WalkResult {
    if (std::optional<std::pair<Value, Value>> moveInfo = getMoveInfo(op)) {
      if (moveInfo->first == moveInfo->second)
        return WalkResult::advance();

      auto srcTy = dyn_cast<RegisterTypeInterface>(moveInfo->first.getType());
      auto tgtTy = dyn_cast<RegisterTypeInterface>(moveInfo->second.getType());
      // Bail if the source or target are not register types, or if they are not
      // the same kind of register type.
      if (!srcTy || !tgtTy || srcTy.getTypeID() != tgtTy.getTypeID())
        return WalkResult::advance();

      // Bail if the source and target are both allocated.
      if (srcTy.hasAllocatedSemantics() && tgtTy.hasAllocatedSemantics())
        return WalkResult::advance();

      if (failed(handleMov(op, *moveInfo)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return std::nullopt;

  // Sort the moves by priority. Stable sort is used to preserve the order of
  // moves with the same priority.
  llvm::stable_sort(movOps, [&](const MovDesc &lhs, const MovDesc &rhs) {
    return lhs.priority < rhs.priority;
  });

  // Optimize the graph.
  int64_t numNodes = graph.sizeNodes();
  IntEquivalenceClasses eqClasses(numNodes);
  optimizeGraph(eqClasses);
  eqClasses.compress();
  if (eqClasses.getNumClasses() == numNodes)
    return std::nullopt;
  return eqClasses;
}

std::optional<IntEquivalenceClasses>
mlir::aster::amdgcn::optimizeGraph(Operation *op,
                                   const RegisterInterferenceGraph &graph,
                                   DataFlowSolver &solver) {
  if (!graph.isCompressed())
    return std::nullopt;
  OptimizeGraphImpl impl(graph, solver);
  return impl.run(op);
}
