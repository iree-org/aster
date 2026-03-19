//===- SchedAttrs.cpp - AMDGCN scheduling attribute implementations -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/AllocaOpInterface.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include <cstdint>

#define DEBUG_TYPE "aster-sched"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// ValueSchedulerAttr - SchedGraphAttrInterface
//===----------------------------------------------------------------------===//

namespace {
struct GraphBuilder {
  GraphBuilder(Block *block, const DataFlowSolver &solver)
      : block(block), solver(const_cast<DataFlowSolver &>(solver)) {
    assert(block && "expected a valid block");
  }

  /// Run the graph builder on the given block, adding edges between operations.
  LogicalResult run(SchedGraph &graph);

private:
  /// Build the SSA dependencies for the graph.
  void buildSSADeps(SchedGraph &graph);

  /// Build the non-SSA dependencies for the graph.
  void buildNonSSADeps(SchedGraph &graph);

  /// Handle a wait operation.
  void handleWaitOp(SchedGraph &graph, int64_t pos, WaitOp wait);

  /// Handle a barrier operation. With barriers we must add dependencies before
  /// and after if the operation affects SALU or SGPRs.
  void handleBarrier(SchedGraph &graph, int64_t pos, Operation *barrier);

  Block *block;
  SmallVector<int64_t> syncPoints;
  DataFlowSolver &solver;
};
} // namespace

LogicalResult
ValueSchedulerAttr::initializeAnalyses(SchedAnalysis &analysis) const {
  // Load the wait analysis.
  analysis.getSolver().load<WaitAnalysis>(analysis.getDomInfo());
  analysis.setRunDataflowAnalyses();
  return success();
}

LogicalResult GraphBuilder::run(SchedGraph &graph) {
  buildSSADeps(graph);
  buildNonSSADeps(graph);
  return success();
}

void GraphBuilder::buildSSADeps(SchedGraph &graph) {
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    LDBG() << "Processing operation: " << i << " "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());

    bool hasEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
                      op->hasTrait<MemoryEffectOpInterface::Trait>();

    // If the operation has no side-effect we need to treat it as a possible
    // sync point. Same for non-pure operations.
    if ((!hasEffects || !mlir::isPure(op)) &&
        !isa<LoadOp, StoreOp, AllocaOpInterface>(op)) {
      LDBG() << "Adding sync point: " << i;
      syncPoints.push_back(i);
    }

    ValueRange deps = op->getOperands();

    // Add edges for the dependencies.
    for (Value operand : deps) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getBlock() == block)
        graph.addEdge(producer, op);
    }
  }
}

void GraphBuilder::buildNonSSADeps(SchedGraph &graph) {
  ArrayRef<Operation *> ops = graph.getOps();
  for (int64_t &i : syncPoints) {
    Operation *op = ops[i];
    if (auto waitOp = dyn_cast<WaitOp>(op)) {
      handleWaitOp(graph, i, waitOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
    if (auto barrierOp = dyn_cast<inst::SOPPOp>(op);
        barrierOp && barrierOp.getOpcode() == amdgcn::OpCode::S_BARRIER) {
      handleBarrier(graph, i, barrierOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
  }

  // Erase all the processed sync points.
  llvm::erase(syncPoints, -1);

  // If there are no sync points, return.
  if (syncPoints.empty())
    return;

  // Add edges between all the ops before a sync point and the sync point, and
  // between the sync point and all the ops after it.
  for (auto [i, syncPoint] : llvm::enumerate(syncPoints)) {
    int64_t prevSyncPoint = i > 0 ? syncPoints[i - 1] : 0;
    int64_t nextSyncPoint =
        i < (syncPoints.size() - 1) ? syncPoints[i + 1] + 1 : ops.size();
    for (int64_t point = prevSyncPoint; point < syncPoint; point++)
      graph.addEdge(point, syncPoint);
    for (int64_t point = syncPoint + 1; point < nextSyncPoint; point++)
      graph.addEdge(syncPoint, point);
  }
}

void GraphBuilder::handleWaitOp(SchedGraph &graph, int64_t pos, WaitOp wait) {
  // Get the wait state.
  const WaitState *state =
      solver.lookupState<WaitState>(solver.getProgramPointAfter(wait));
  assert(state && "expected valid wait state");

  // Collect all the operations that sync at this point.
  SetVector<Operation *> waitedOps;
  for (const TokenState &token : state->waitOpInfo->waitedTokens) {
    Operation *op = token.getToken().getDefiningOp();
    if (!op || op->getBlock() != block)
      continue;
    waitedOps.insert(op);
  }
  for (const TokenState &token : state->waitOpInfo->impliedTokens) {
    Operation *op = token.getToken().getDefiningOp();
    if (!op || op->getBlock() != block)
      continue;
    waitedOps.insert(op);
  }

  // Add edges for the waited operations.
  for (Operation *op : waitedOps)
    graph.addEdge(op, wait);

  auto isTokenKind = [](Type type, MemoryInstructionKind kind) {
    if (auto tokType = dyn_cast<ReadTokenType>(type))
      return tokType.getKind() == kind;
    if (auto tokType = dyn_cast<WriteTokenType>(type))
      return tokType.getKind() == kind;
    return false;
  };

  // Add edges for the operations that are after this wait operation that wait
  // on the same tokens.
  bool waitsVM = wait.getVmCnt() != WaitOp::kNoWaitCount;
  bool waitsLgkm = wait.getLgkmCnt() != WaitOp::kNoWaitCount;
  for (Operation *op : graph.getOps().drop_front(pos + 1)) {
    ValueRange operands = op->getOperands();
    for (Value operand : operands) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || producer->getBlock() != block)
        continue;
      if (waitedOps.contains(producer))
        graph.addEdge(wait, op);
    }

    // Prevent operations producing tokens moving before a wait operation.
    bool producesVM = llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
      return isTokenKind(type, MemoryInstructionKind::Flat);
    });
    bool producesLgkm =
        llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
          return isTokenKind(type, MemoryInstructionKind::Shared) ||
                 isTokenKind(type, MemoryInstructionKind::Constant);
        });
    if (waitsVM && producesVM)
      graph.addEdge(wait, op);
    if (waitsLgkm && producesLgkm)
      graph.addEdge(wait, op);
  }
}

void GraphBuilder::handleBarrier(SchedGraph &graph, int64_t pos,
                                 Operation *barrier) {
  // Helper function to add an edge between an operation and the barrier.
  auto addEdge = [&](Operation *op, int64_t i) {
    if (i < pos)
      graph.addEdge(op, barrier);
    if (i > pos)
      graph.addEdge(barrier, op);
  };

  // Iterate over all the operations in the graph.
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    // Skip itself.
    if (op == barrier)
      continue;

    auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
    const InstMetadata *metadata = instOp ? instOp.getInstMetadata() : nullptr;

    // If there's no metadata, add an edge from the barrier to the operation.
    if (!metadata) {
      if (!isPure(op))
        addEdge(op, i);
      continue;
    }

    // If the operation has any SALU or SMEM properties, add an edge.
    if (metadata->hasAnyProps({InstProp::Salu, InstProp::Smem})) {
      addEdge(op, i);
      continue;
    }

    // If the operation has any SGPR outputs, add an edge.
    bool hasSGPROut = llvm::any_of(instOp->getResults(), [](Value result) {
      return isa<SGPRType>(result.getType());
    });
    if (hasSGPROut)
      addEdge(op, i);
  }
}

FailureOr<SchedGraph>
ValueSchedulerAttr::createGraph(Block *block,
                                const SchedAnalysis &analysis) const {
  SchedGraph graph(block);
  GraphBuilder builder(block, analysis.getSolver());
  if (failed(builder.run(graph)))
    return failure();
  graph.compress();
  return graph;
}

//===----------------------------------------------------------------------===//
// InstPropLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t InstPropLabelerAttr::getLabel(Operation *op, int32_t,
                                      const SchedGraph &) const {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  const InstMetadata *metadata = instOp ? instOp.getInstMetadata() : nullptr;
  if (!metadata)
    return -1;
  ArrayRef<InstProp> matcher = getInstMatcher();
  if (matcher.empty())
    return getStage();
  if (!metadata->hasAnyProps(matcher))
    return -1;
  return getStage();
}

//===----------------------------------------------------------------------===//
// OpCodeLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t OpCodeLabelerAttr::getLabel(Operation *op, int32_t,
                                    const SchedGraph &) const {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  const InstMetadata *metadata = instOp ? instOp.getInstMetadata() : nullptr;
  if (!metadata)
    return -1;
  ArrayRef<OpCode> matcher = getOpCodeMatcher();
  if (matcher.empty())
    return getStage();
  OpCode opcode = metadata->getOpCode();
  if (!llvm::is_contained(matcher, opcode))
    return -1;
  return getStage();
}

//===----------------------------------------------------------------------===//
// LatencyPipelinerSchedAttr - SchedBuilderAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult
LatencyPipelinerSchedAttr::createSched(const SchedGraph &schedGraph,
                                       SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  int32_t limit = getSchedLimit();
  if (limit <= 0)
    limit = std::numeric_limits<int32_t>::max();

  auto schedFn = [&](ArrayRef<int32_t> ready, SetVector<int32_t> &indices) {
    bool hasZeroLabel = false;
    // Phase 1: schedule label-0 ops immediately.
    for (auto [i, node] : llvm::enumerate(ready)) {
      if (schedGraph.getLabel(node) == 0) {
        hasZeroLabel = true;
        indices.insert(i);
      }
    }

    // Early exit if we scheduled any label-0 ops.
    if (hasZeroLabel)
      return;

    // Phase 2: interleaved highest-latency scheduling.
    using IntPair = std::pair<int32_t, int32_t>;
    SmallVector<IntPair> sortedReady;
    sortedReady =
        llvm::map_to_vector(llvm::enumerate(ready), [&](auto indexNode) {
          auto [i, node] = indexNode;
          return std::make_pair(node, static_cast<int32_t>(i));
        });
    llvm::sort(sortedReady, [&](IntPair a, IntPair b) {
      int32_t labelA = schedGraph.getLabel(a.first);
      int32_t labelB = schedGraph.getLabel(b.first);
      // Sort descending by label; break ties by ascending node ID.
      if (labelA != labelB)
        return labelA > labelB;
      return a.first < b.first;
    });
    if (sortedReady.empty())
      return;

    int32_t numToAdd = std::min<int32_t>(limit, ready.size());
    int32_t currLabel = schedGraph.getLabel(sortedReady.front().first);
    int32_t idx = 1;
    // Schedule the highest-latency op.
    indices.insert(sortedReady.front().second);
    --numToAdd;

    // Circular scheduler, always trying to schedule ops with different labels.
    while (numToAdd > 0) {
      // Skip already-inserted ops and ops with the same label as the last
      // scheduled one, advancing the circular index.
      if (indices.contains(sortedReady[idx].second) ||
          currLabel == schedGraph.getLabel(sortedReady[idx].first)) {
        idx = (idx + 1) % sortedReady.size();

        // If we looped, reset the label so in the next iteration we will
        // schedule the highest-latency op among the remaining ones.
        if (idx == 0)
          currLabel = std::numeric_limits<int32_t>::min();
        continue;
      }
      // Schedule the op and update the current label.
      indices.insert(sortedReady[idx].second);
      --numToAdd;
      currLabel = schedGraph.getLabel(sortedReady[idx].first);
      idx = (idx + 1) % sortedReady.size();
      if (idx == 0)
        currLabel = std::numeric_limits<int32_t>::min();
    }
  };

  return schedGraph.topologicalSched(schedFn, sched);
}
