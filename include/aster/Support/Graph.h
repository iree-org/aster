//===- Graph.h - Graph utilities ---------------------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_GRAPH_H_H
#define ASTER_SUPPORT_GRAPH_H_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::aster {

/// Graph data structure supporting directed and undirected graphs. Nodes are
/// identified by integer IDs, and are expected to be dense, contiguous and
/// starting on 0.
struct Graph {
  using NodeID = int32_t;
  using Edge = std::pair<NodeID, NodeID>;
  using NodeIterator = llvm::detail::index_iterator;
  using EdgeIterator = typename SmallVector<Edge>::const_iterator;

  Graph(bool directed) : directed(directed) {}

  /// Check if the graph is directed.
  bool isDirected() const { return directed; }

  /// Check if the graph is compressed.
  bool isCompressed() const { return compressed; }

  /// Compress the graph representation.
  void compress() {
    llvm::sort(edgeVector);
    compressed = true;
    node2Edge.assign(numNodes + 1, 0);
    for (const Edge &edge : edgeVector)
      ++node2Edge[edge.first + 1];
    for (int i = 1, end = node2Edge.size(); i < end; ++i)
      node2Edge[i] += node2Edge[i - 1];
  }

  /// Add an edge to the graph.
  void addEdge(NodeID src, NodeID tgt) {
    bool inserted = insertEdge(src, tgt);
    if (!directed && inserted)
      insertEdge(tgt, src);
  }

  /// Get the number of nodes in the graph.
  int sizeNodes() const { return numNodes; }

  /// Get the number of edges in the graph.
  int sizeEdges() const {
    if (directed)
      return edgeVector.size();
    return edgeVector.size() / 2;
  }

  /// Check if the graph has a given node.
  bool hasNode(NodeID node) const { return node >= 0 && node < numNodes; }

  /// Check if the graph has an edge between src and tgt.
  bool hasEdge(NodeID src, NodeID tgt) const {
    return edgeSet.contains({src, tgt});
  }

  /// Get the range of nodes in the graph.
  llvm::index_range nodes() const {
    assert(compressed && "Graph must be compressed to iterate nodes");
    return llvm::index_range(0, numNodes);
  }

  /// Get the begin iterator for nodes.
  NodeIterator nodesBegin() const {
    assert(compressed && "Graph must be compressed to iterate nodes");
    return NodeIterator(0);
  }
  /// Get the end iterator for nodes.
  NodeIterator nodesEnd() const {
    assert(compressed && "Graph must be compressed to iterate nodes");
    return NodeIterator(numNodes);
  }

  /// Get the begin iterator for edges.
  EdgeIterator edgesBegin() const {
    assert(compressed && "Graph must be compressed to iterate edges");
    return edgeVector.begin();
  }
  /// Get the end iterator for edges.
  EdgeIterator edgesEnd() const {
    assert(compressed && "Graph must be compressed to iterate edges");
    return edgeVector.end();
  }

  /// Get the range of edges in the graph.
  auto edges() const { return llvm::make_range(edgesBegin(), edgesEnd()); }

  /// Get the edges for a specific node.
  auto edges(int node) const {
    assert(compressed && "Graph must be compressed to iterate edges");
    assert(node >= 0 && node < numNodes && "Node ID out of range");
    return llvm::make_range(edgesBegin() + node2Edge[node],
                            edgesBegin() + node2Edge[node + 1]);
  }

  /// Set the number of nodes in the graph.
  void setNumNodes(int n) {
    assert(
        numNodes <= n &&
        "New number of nodes must be greater or equal to the current number");
    numNodes = n;
    compressed = false;
  }

  /// Get the array mapping nodes to edges.
  ArrayRef<int64_t> getNode2Edges() const { return node2Edge; }

  /// Get the range of edges for a specific node.
  std::pair<int64_t, int64_t> getEdgesPos(int ic) const {
    assert(ic < numNodes);
    assert(compressed && "Graph must be compressed");
    return {node2Edge[ic], node2Edge[ic + 1]};
  }

  /// Get the array of edges.
  ArrayRef<Edge> getEdges() const { return edgeVector; }

  /// Get a specific edge by index.
  Edge getEdge(int index) const {
    assert(index >= 0 && static_cast<size_t>(index) < edgeVector.size() &&
           "Edge index out of range");
    return edgeVector[index];
  }

  /// Print the graph.
  void print(
      raw_ostream &os, llvm::StringRef name = "Graph",
      function_ref<void(raw_ostream &os, const NodeID &)> nodePrinter = {},
      function_ref<void(raw_ostream &os, const Edge &)> edgePrinter = {}) const;

  /// Computes the in degree of each node.
  SmallVector<int32_t> getInDegree() const;

  /// Compute topological sort of the graph. Returns the sorted nodes or
  /// failure if the graph contains a cycle. Only works for directed graphs.
  FailureOr<SmallVector<NodeID>> topologicalSort() const;

protected:
  /// Insert a directed edge.
  bool insertEdge(NodeID src, NodeID tgt) {
    assert(src >= 0 && tgt >= 0 && "Node IDs must be non-negative");
    numNodes = std::max({numNodes, src, tgt + 1});
    if (!edgeSet.insert({src, tgt}).second)
      return false;
    edgeVector.push_back({src, tgt});
    compressed = false;
    return true;
  }

  SmallVector<int64_t> node2Edge;
  SmallVector<Edge> edgeVector;
  DenseSet<Edge> edgeSet;
  int numNodes = 0;
  bool compressed = true;
  bool directed = false;
};
} // end namespace mlir::aster

#endif // ASTER_SUPPORT_GRAPH_H_H
