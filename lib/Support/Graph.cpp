//===- Graph.cpp - Graph utilities ----------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Support/Graph.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::aster;

/// Default node printer.
static void defaultNodePrinter(raw_ostream &os, const Graph::NodeID &node) {
  os << "label = \"" << node << "\"";
}

/// Default edge printer.
static void defaultEdgePrinter(raw_ostream &os, const Graph::Edge &edge) {
  os << "label = \"" << edge.first << ", " << edge.second << "\"";
}

void Graph::print(
    raw_ostream &os, llvm::StringRef name,
    function_ref<void(raw_ostream &os, const NodeID &)> nodePrinter,
    function_ref<void(raw_ostream &os, const Edge &)> edgePrinter) const {
  assert(compressed && "Graph must be compressed before printing");
  if (!nodePrinter)
    nodePrinter = defaultNodePrinter;
  if (!edgePrinter)
    edgePrinter = defaultEdgePrinter;
  os << (isDirected() ? "digraph " : "graph ") << name.str() << " {\n";
  llvm::interleave(
      nodes(), os,
      [&](NodeID node) {
        os << "  " << node << " [";
        nodePrinter(os, node);
        os << "];";
      },
      "\n");
  os << "\n";
  StringRef edgeKind = isDirected() ? " -> " : " -- ";
  for (const Edge &edge : edges()) {
    NodeID src = edge.first;
    NodeID tgt = edge.second;
    if (!isDirected() && src > tgt)
      continue;
    os << "  " << src << edgeKind << tgt << " [";
    edgePrinter(os, edge);
    os << "];\n";
  }
  os << "}";
}

SmallVector<int32_t> Graph::getInDegree() const {
  // Count in-degrees for each node
  SmallVector<int32_t> inDegree(numNodes, 0);
  for (const Edge &edge : edges())
    ++inDegree[edge.second];
  return inDegree;
}

FailureOr<SmallVector<Graph::NodeID>> Graph::topologicalSort() const {
  assert(compressed && "Graph must be compressed before topological sort");
  assert(isDirected() && "Topological sort only works for directed graphs");

  SmallVector<NodeID> result;
  result.reserve(numNodes);

  // Count in-degrees for each node
  SmallVector<int32_t> inDegree = getInDegree();

  // Queue of nodes with in-degree 0
  SmallVector<NodeID> queue;
  for (NodeID node : nodes()) {
    if (inDegree[node] == 0)
      queue.push_back(node);
  }

  // Process nodes with in-degree 0
  while (!queue.empty()) {
    NodeID node = queue.pop_back_val();
    result.push_back(node);

    // Reduce in-degree for neighbors
    for (const Edge &edge : edges(node)) {
      NodeID neighbor = edge.second;
      --inDegree[neighbor];
      if (inDegree[neighbor] == 0)
        queue.push_back(neighbor);
    }
  }

  // Check if all nodes were processed (no cycle)
  if (result.size() != static_cast<size_t>(numNodes))
    return failure();

  return result;
}
