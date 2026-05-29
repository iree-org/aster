//===- OptimizeInterferenceGraph.cpp - Optimize interference graph --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/InstImpl.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/DebugLog.h"

#include <limits>

#define DEBUG_TYPE "amdgcn-interference-opt"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
using EqClasses = llvm::EquivalenceClasses<int32_t>;

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
  OptimizeGraphImpl(RegisterInterferenceGraph &graph, EqClasses &eqClasses,
                    llvm::IntEqClasses &nodeClasses)
      : graph(graph), eqClasses(eqClasses), nodeClasses(nodeClasses) {}

  /// Build equivalence classes for the interference graph.
  LogicalResult run(Operation *root);

  /// Collect a move operation: resolve its allocas and assign a coalescing
  /// priority (0 = load-sourced, preferred; 1 = default).
  LogicalResult collectMov(Operation *op, std::pair<Value, Value> moveInfo);

  /// Optimize the graph and populate the equivalence classes.
  void optimizeGraph();

  RegisterInterferenceGraph &graph;
  EqClasses &eqClasses;
  llvm::IntEqClasses &nodeClasses;
  SmallVector<MovDesc> movOps;
};
} // namespace

static FailureOr<std::pair<Value, Value>> getMoveInfo(Operation *op) {
  if (auto copyOp = dyn_cast<lsir::CopyOp>(op))
    return std::pair<Value, Value>(copyOp.getSource(), copyOp.getTarget());
  if (auto vop1 = dyn_cast<VMovB32>(op))
    return std::pair<Value, Value>(vop1.getSrc0(), vop1.getDst0());
  if (auto sop1 = dyn_cast<SMovB32>(op))
    return std::pair<Value, Value>(sop1.getSrc0(), sop1.getDst0());
  return failure();
}

LogicalResult OptimizeGraphImpl::collectMov(Operation *op,
                                            std::pair<Value, Value> moveInfo) {
  FailureOr<ValueRange> srcAlloc = getAllocasOrFailure(moveInfo.first);
  if (failed(srcAlloc))
    return failure();
  FailureOr<ValueRange> tgtAlloc = getAllocasOrFailure(moveInfo.second);
  if (failed(tgtAlloc))
    return failure();

  // Bias coalescing toward MOVs whose source was just loaded from memory.
  int32_t priority =
      llvm::any_of(*srcAlloc,
                   [&](Value v) { return hasReachingLoadDefinition(op, v); })
          ? 0
          : 1;

  movOps.push_back({op, *srcAlloc, *tgtAlloc, priority});
  return success();
}

/// Check if the given classes have an edge between them.
static bool hasEdge(const Graph &graph, EqClasses &eqClasses, int32_t lhsNode,
                    int32_t rhsNode) {
  int32_t rhsLeader = eqClasses.getLeaderValue(rhsNode);
  for (int32_t member : eqClasses.members(lhsNode)) {
    for (auto [src, tgt] : graph.edges(member)) {
      if (eqClasses.getLeaderValue(tgt) == rhsLeader)
        return true;
    }
  }
  return false;
}

/// Check if the given classes can be coalesced.
static bool canCoalesce(const RegisterInterferenceGraph &graph,
                        EqClasses &eqClasses, int32_t src, int32_t tgt,
                        int32_t size) {
  LDBG() << "-- Checking if can coalesce: " << src << " and " << tgt
         << " with size " << size;
  for (auto [srcId, tgtId] :
       llvm::zip_equal(llvm::seq<int32_t>(src, src + size),
                       llvm::seq<int32_t>(tgt, tgt + size))) {
    // If the nodes are the same continue.
    if (srcId == tgtId)
      continue;

    // Continue if the classes are the same.
    if (eqClasses.isEquivalent(srcId, tgtId))
      continue;

    // Check if the classes have an edge between them.
    if (hasEdge(graph, eqClasses, srcId, tgtId)) {
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
/// one, or if they have the same size, to the one with the largest alignment.
/// Otherwise, we would need to create a new range containing both ranges and
/// destroy the current ranges, which also implies a re-numbering of the nodes
/// in the graph.
/// TODO: Allow merging with re-numbering.
static FailureOr<std::tuple<int32_t, int32_t, int32_t, int32_t>>
getRangeBounds(int32_t lhsBegin, int32_t lhsOff, int32_t lhsSize,
               int32_t lhsAlignment, int32_t rhsBegin, int32_t rhsOff,
               int32_t rhsSize, int32_t rhsAlignment) {
  // Swap so that the lhs is the larger range.
  if (lhsSize < rhsSize)
    return getRangeBounds(rhsBegin, rhsOff, rhsSize, rhsAlignment, lhsBegin,
                          lhsOff, lhsSize, lhsAlignment);

  // Get the start of the rhs in the lhs according to the copy offset.
  int32_t rhsInLhsStart = lhsOff - rhsOff;

  // Bail if we cannot fit the elements before the rhs offset in the lhs range.
  if (rhsInLhsStart < 0)
    return failure();

  // Bail if we cannot fit the elements after the rhs offset in the lhs range.
  if (rhsInLhsStart + rhsSize > lhsSize)
    return failure();

  assert(lhsAlignment > 0 && "alignment must be positive");
  assert(rhsAlignment > 0 && "alignment must be positive");

  // Bail if the start position of rhs side doesn't satisfy it's alignment
  // requirement.
  if (rhsInLhsStart % rhsAlignment != 0)
    return failure();

  // Get the new alignment for the coalesced range.
  int32_t newAlignment = std::lcm(lhsAlignment, rhsAlignment);

  return std::make_tuple(static_cast<int32_t>(rhsInLhsStart + lhsBegin),
                         rhsBegin, rhsSize, newAlignment);
}

/// Optimize the graph by coalescing the registers involved in the moves.
/// NOTE: Given the presence of register ranges, it is not enough to only check
/// the registers involved in the moves; we need to check that one of the entire
/// ranges can be coalesced into the other.
void OptimizeGraphImpl::optimizeGraph() {
  for (const MovDesc &mov : movOps) {
    LDBG() << "Processing move: " << *mov.moveOp;

    // NOTE: This is safe because the graph provides the guarantee that ranges
    // are consecutive.
    int32_t srcId = nodeClasses.findLeader(graph.getNodeId(mov.srcAllocas[0]));
    int32_t tgtId =
        nodeClasses.findLeader(graph.getNodeId(mov.targetAllocas[0]));
    LDBG() << "- Source ID: " << srcId << ", Target ID: " << tgtId;

    // If srcId == tgtId, this is a trivial self-copy: continue.
    if (srcId == tgtId)
      continue;

    auto [srcLeaderId, srcRange] = graph.getRangeInfo(srcId);
    auto [tgtLeaderId, tgtRange] = graph.getRangeInfo(tgtId);
    LDBG() << "- Source leader range ID: " << srcLeaderId
           << ", Target leader range ID: " << tgtLeaderId;

    // Get the start of the copy offsets.
    int32_t srcOffset = srcId - srcLeaderId;
    int32_t tgtOffset = tgtId - tgtLeaderId;
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
    FailureOr<std::tuple<int32_t, int32_t, int32_t, int32_t>> bounds =
        getRangeBounds(srcLeaderId, srcOffset, srcRangeSize, srcRangeAlignment,
                       tgtLeaderId, tgtOffset, tgtRangeSize, tgtRangeAlignment);
    if (failed(bounds)) {
      LDBG() << "--- Ranges are incompatible";
      continue;
    }

    auto [lhsStart, rhsStart, size, newAlignment] = *bounds;
    LDBG() << "- Left start: " << lhsStart << ", Right start: " << rhsStart
           << ", Size: " << size;

    // Continue if the ranges cannot be coalesced.
    if (!canCoalesce(graph, eqClasses, lhsStart, rhsStart, size))
      continue;

    // Coalesce the ranges.
    for (auto [srcId, tgtId] :
         llvm::zip_equal(llvm::seq<int32_t>(lhsStart, lhsStart + size),
                         llvm::seq<int32_t>(rhsStart, rhsStart + size))) {
      eqClasses.unionSets(srcId, tgtId);
      nodeClasses.join(srcId, tgtId);
      LDBG() << "--- Joined: " << srcId << " and " << tgtId;
    }

    // Update the alignment of the ranges.
    if (srcRange)
      srcRange->alignment = newAlignment;
    if (tgtRange)
      tgtRange->alignment = newAlignment;
  }
}

LogicalResult OptimizeGraphImpl::run(Operation *root) {
  WalkResult result = root->walk([&](Operation *op) -> WalkResult {
    FailureOr<std::pair<Value, Value>> moveInfo = getMoveInfo(op);
    if (failed(moveInfo))
      return WalkResult::advance();

    if (moveInfo->first == moveInfo->second)
      return WalkResult::advance();

    auto srcTy = dyn_cast<RegisterTypeInterface>(moveInfo->first.getType());
    auto tgtTy = dyn_cast<RegisterTypeInterface>(moveInfo->second.getType());
    if (!srcTy || !tgtTy || srcTy.getTypeID() != tgtTy.getTypeID())
      return WalkResult::advance();

    if (srcTy.hasAllocatedSemantics() && tgtTy.hasAllocatedSemantics())
      return WalkResult::advance();

    if (failed(collectMov(op, *moveInfo)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Sort the moves by priority. Stable sort is used to preserve the order of
  // moves with the same priority.
  llvm::stable_sort(movOps, [&](const MovDesc &lhs, const MovDesc &rhs) {
    return lhs.priority < rhs.priority;
  });

  // Optimize the graph.
  int64_t numNodes = graph.sizeNodes();
  nodeClasses.grow(numNodes);
  for (int32_t i = 0; i < numNodes; ++i)
    eqClasses.insert(i);
  optimizeGraph();
  nodeClasses.compress();
  return success();
}

/// Build a CoalescingInfo from the interference graph and equivalence classes.
/// nodeClasses must already be compressed. For each quotient node the
/// representative is the minimum-ID original node in that class, which is
/// always the range leader because ranges have consecutive IDs by construction.
static CoalescingInfo buildCoalescingInfo(RegisterInterferenceGraph &graph,
                                          llvm::IntEqClasses &nodeClasses) {
  int32_t numQuotient = static_cast<int32_t>(nodeClasses.getNumClasses());
  int32_t numNodes = graph.sizeNodes();
  CoalescingInfo info;

  // Build the forward map (nodeClass) and count class sizes for the CSR
  // offset array.
  info.nodeClass.resize(numNodes);
  info.memberOffsets.resize(numQuotient + 1, 0);
  for (int32_t i = 0; i < numNodes; ++i) {
    int32_t qid = static_cast<int32_t>(nodeClasses[i]);
    info.nodeClass[i] = qid;
    ++info.memberOffsets[qid + 1];
  }
  // Prefix-sum the sizes to get offsets.
  for (int32_t qid = 0; qid < numQuotient; ++qid)
    info.memberOffsets[qid + 1] += info.memberOffsets[qid];

  // Fill memberData in one pass. Nodes are visited in ascending order so each
  // class's slice is already sorted and its first element is the
  // representative.
  info.memberData.resize(numNodes);
  SmallVector<int32_t> cursor(info.memberOffsets.begin(),
                              info.memberOffsets.begin() + numQuotient);
  for (int32_t i = 0; i < numNodes; ++i) {
    int32_t qid = info.nodeClass[i];
    info.memberData[cursor[qid]++] = i;
  }

  // Populate values and constraints using the minimum member of each class as
  // the representative.
  info.values.resize(numQuotient);
  info.constraints.resize(numQuotient);
  for (int32_t qid = 0; qid < numQuotient; ++qid) {
    int32_t leader = info.memberData[info.memberOffsets[qid]];
    info.values[qid] = graph.getValue(leader);
    auto [rangeId, constraint] = graph.getRangeInfo(leader);
    // Only store the constraint when the leader is also the range leader so
    // that allocation iterates the range exactly once per quotient node.
    info.constraints[qid] = (rangeId == leader) ? constraint : nullptr;
  }

  // Build the quotient graph by remapping edges directly, without copying the
  // original edge set into an intermediate graph first.
  info.graph = Graph(/*directed=*/false);
  info.graph.setNumNodes(numQuotient);
  for (auto [src, tgt] : graph.edges()) {
    int32_t s = static_cast<int32_t>(nodeClasses[src]);
    int32_t t = static_cast<int32_t>(nodeClasses[tgt]);
    if (s != t)
      info.graph.addEdge(s, t);
  }
  info.graph.compress();

  // Verify: each quotient slot contains at most one physical register.
  // Coalescing skips moves where both sides are allocated, but transitive
  // merges through an unallocated intermediate could still violate this.
  // Defensively assert for now but may require fixing in the future.
  for (int32_t qid = 0; qid < numQuotient; ++qid) {
    std::optional<AMDGCNRegisterTypeInterface> seenFixed;
    for (int32_t j = info.memberOffsets[qid], end = info.memberOffsets[qid + 1];
         j < end; ++j) {
      auto regTy = cast<AMDGCNRegisterTypeInterface>(
          graph.getValue(info.memberData[j]).getType());
      if (!regTy.hasAllocatedSemantics())
        continue;
      assert((!seenFixed || *seenFixed == regTy) &&
             "coalescing class contains multiple preallocated registers");
      seenFixed = regTy;
    }
  }

  return info;
}

FailureOr<CoalescingInfo>
CoalescingInfo::optimizeGraph(Operation *op, RegisterInterferenceGraph &graph) {
  assert(graph.isCompressed() && "graph must be compressed");
  EqClasses eqClasses;
  llvm::IntEqClasses nodeClasses;
  if (failed(OptimizeGraphImpl(graph, eqClasses, nodeClasses).run(op)))
    return failure();
  return buildCoalescingInfo(graph, nodeClasses);
}
