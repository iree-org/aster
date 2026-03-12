//===- SchedInterfaces.cpp - Scheduling attribute interfaces -----*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the scheduling attribute interfaces.
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "aster-sched"

using namespace mlir;
using namespace mlir::aster;

void SchedGraph::initialize() {
  for (Operation &op : block->getOperations()) {
    // Skip terminators.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    opToId[&op] = ops.size();
    ops.push_back(&op);
  }
  labels.resize(ops.size(), -1);
  setNumNodes(ops.size());
}

LogicalResult
SchedGraph::topologicalSched(function_ref<int32_t(ArrayRef<int32_t>)> schedFn,
                             SmallVectorImpl<int32_t> &sched) const {
  assert(compressed && "Graph must be compressed before topological sort");

  sched.clear();
  sched.reserve(numNodes);

  // Count in-degrees for each node
  SmallVector<int32_t> inDegree = getInDegree();

  LDBG() << "In-degree: " << llvm::interleaved_array(inDegree);

  // Queue of nodes with in-degree 0
  SmallVector<int32_t> queue;
  for (int32_t node : nodes()) {
    if (inDegree[node] == 0)
      queue.push_back(node);
  }

  // Process nodes with in-degree 0
  while (!queue.empty()) {
    LDBG() << "Queue: " << llvm::interleaved_array(queue);
    int64_t nextNodePos = schedFn(queue);
    assert(nextNodePos >= 0 &&
           nextNodePos < static_cast<int64_t>(queue.size()) &&
           "Invalid next node position");
    int32_t nextNode = queue[nextNodePos];
    queue.erase(queue.begin() + nextNodePos);
    sched.push_back(nextNode);

    LDBG() << "Selected node: " << nextNode << " (position: " << nextNodePos
           << ")";

    // Reduce in-degree for neighbors
    for (const Edge &edge : edges(nextNode)) {
      int32_t neighbor = edge.second;
      --inDegree[neighbor];
      if (inDegree[neighbor] == 0)
        queue.push_back(neighbor);
    }
  }

  // Check if all nodes were processed (no cycle)
  if (sched.size() != static_cast<size_t>(numNodes)) {
    LDBG() << "Failed to produce topological sort order: " << sched.size()
           << " != " << numNodes;
    return failure();
  }

  return success();
}

void SchedGraph::applySched(const SchedGraph &schedGraph,
                            RewriterBase &rewriter, ArrayRef<int32_t> order) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *block = schedGraph.getBlock();
  Block::iterator lastSched = block->begin();
  for (int32_t nodeId : order) {
    rewriter.setInsertionPoint(block, lastSched);
    Operation *op = schedGraph.getOp(nodeId);
    rewriter.moveOpAfter(op, block, lastSched);
    lastSched = op->getIterator();
  }
}

#include "aster/Interfaces/SchedInterfaces.cpp.inc"
