//===- LDSInterferenceGraph.h - LDS Interference Graph -----------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LDS interference graph for analyzing LDS buffer
// allocations. The graph tracks which buffers are simultaneously live and
// therefore cannot share memory.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_LDSINTERFERENCEGRAPH_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_LDSINTERFERENCEGRAPH_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Support/Graph.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::aster::amdgcn {

//===----------------------------------------------------------------------===//
// LDSAllocNode
//===----------------------------------------------------------------------===//

/// A node in the LDS interference graph, representing an AllocLDSOp.
struct LDSAllocNode {
  AllocLDSOp allocOp;
  int64_t size = 0;
  int64_t alignment = 0;

  LDSAllocNode(AllocLDSOp op, int64_t size, int64_t alignment)
      : allocOp(op), size(size), alignment(alignment) {}
};

//===----------------------------------------------------------------------===//
// LDSInterferenceGraph
//===----------------------------------------------------------------------===//

/// An interference graph for LDS allocations. Nodes are AllocLDSOp operations,
/// and edges connect buffers that are simultaneously live at some program
/// point.
///
/// The graph is built by:
/// 1. Collecting all AllocLDSOp operations as nodes
/// 2. Running buffer analysis to track liveness
/// 3. Adding edges between buffers that are live at the same program point
class LDSInterferenceGraph : public Graph {
public:
  using NodeId = Graph::NodeID;

  LDSInterferenceGraph() : Graph(/*directed=*/false) {}

  /// Create the interference graph from an operation.
  /// Returns failure if:
  /// - Any buffer has a `top` liveness state (conflicting information)
  /// - Any `get_lds_offset` operation acts on a dead buffer
  /// - Any allocation has a non-constant size
  static FailureOr<LDSInterferenceGraph> create(Operation *op,
                                                DominanceInfo &domInfo);

  /// Get the node ID for a buffer value. Returns -1 if not found.
  NodeId getNodeId(Value buffer) const;

  /// Get the allocation nodes in the graph.
  ArrayRef<LDSAllocNode> getAllocNodes() const { return allocNodes; }

  /// Print the graph in DOT format.
  void print(raw_ostream &os) const;

private:
  /// Add an allocation node to the graph. Returns the node ID.
  NodeId addNode(AllocLDSOp allocOp, int64_t size, int64_t alignment);

  /// Verify liveness constraints and build interference edges.
  LogicalResult buildAndVerify(Operation *op, DataFlowSolver &solver);

  SmallVector<LDSAllocNode> allocNodes;
  llvm::DenseMap<Value, NodeId> allocToNodeId;
};

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_LDSINTERFERENCEGRAPH_H
