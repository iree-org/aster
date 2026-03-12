//===- SchedInterfaces.h - Scheduling attribute interfaces -------*- C++
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
// This file declares the scheduling attribute interfaces and SchedGraph class.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_SCHEDINTERFACES_H
#define ASTER_INTERFACES_SCHEDINTERFACES_H

#include "aster/Support/Graph.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
class DataFlowSolver;
class DominanceInfo;

namespace aster {
/// Helper class to collect the analysis required for scheduling.
class SchedAnalysis {
public:
  SchedAnalysis(Operation *rootOp, DataFlowSolver &solver,
                DominanceInfo &domInfo, AnalysisManager analysisManager)
      : rootOp(rootOp), solver(solver), domInfo(domInfo),
        analysisManager(analysisManager) {
    assert(rootOp && "expected a valid operation");
  }

  /// Get the data flow solver.
  DataFlowSolver &getSolver() { return solver; }
  const DataFlowSolver &getSolver() const { return solver; }

  /// Get the dominance information.
  DominanceInfo &getDomInfo() { return domInfo; }
  const DominanceInfo &getDomInfo() const { return domInfo; }

  /// Get the root operation.
  Operation *getRootOp() const { return rootOp; }

  /// Get the analysis manager.
  AnalysisManager getAnalysisManager() const { return analysisManager; }

  /// Set whether to run dataflow analyses.
  void setRunDataflowAnalyses() { runDataflowAnalyses = true; }

  /// Check if dataflow analyses should be run.
  bool shouldRunDataflowAnalyses() const { return runDataflowAnalyses; }

private:
  Operation *rootOp;
  DataFlowSolver &solver;
  DominanceInfo &domInfo;
  AnalysisManager analysisManager;
  bool runDataflowAnalyses = false;
};

/// Scheduling DAG graph that extends the base Graph with block-specific data.
/// Contains the operations in the block, a pointer to the owner block, and
/// labels for each node.
class SchedGraph : public Graph {
public:
  explicit SchedGraph(Block *block) : Graph(/*directed=*/true), block(block) {
    assert(block && "expected a valid block");
    initialize();
  }

  /// Get the owner block.
  Block *getBlock() const { return block; }

  /// Get the operations in the block in order. The index of an operation in
  /// this array corresponds to its node ID.
  llvm::ArrayRef<Operation *> getOps() const { return ops; }

  /// Get the labels for each node.
  llvm::ArrayRef<int32_t> getLabels() const { return labels; }

  /// Get the node ID for an operation. Returns -1 if the operation is not in
  /// the graph.
  int64_t getOpId(Operation *op) const { return opToId.lookup_or(op, -1); }

  /// Get the operation for a node ID. Returns nullptr if the node ID is not in
  /// the graph.
  Operation *getOp(int64_t id) const {
    return id >= 0 && id < static_cast<int64_t>(ops.size()) ? ops[id] : nullptr;
  }

  /// Get the label for a node ID. Asserts if the node ID is not in the graph.
  int32_t getLabel(int64_t id) const { return labels[id]; }

  /// Set the label for a node ID. Asserts if the node ID is not in the graph.
  void setLabel(int64_t id, int32_t label) { labels[id] = label; }

  /// Add an edge to the graph.
  void addEdge(Operation *src, Operation *tgt) {
    int64_t srcId = getOpId(src);
    int64_t tgtId = getOpId(tgt);
    if (srcId == -1 || tgtId == -1)
      return;
    Graph::addEdge(srcId, tgtId);
  }
  using Graph::addEdge;

  /// Check if the graph has an edge between src and tgt.
  bool hasEdge(Operation *src, Operation *tgt) const {
    int64_t srcId = getOpId(src);
    int64_t tgtId = getOpId(tgt);
    if (srcId == -1 || tgtId == -1)
      return false;
    return Graph::hasEdge(srcId, tgtId);
  }
  using Graph::hasEdge;

  /// Produce a topologically sorted schedule for the graph using the given
  /// scheduling function.
  /// The scheduling function receives a list of node IDs that are ready to be
  /// scheduled, and returns the position of the next node to schedule relative
  /// to the ready nodes.
  LogicalResult
  topologicalSched(function_ref<int32_t(ArrayRef<int32_t>)> schedFn,
                   llvm::SmallVectorImpl<int32_t> &sched) const;

  /// Apply the schedule to the graph using the given rewriter and order.
  static void applySched(const SchedGraph &schedGraph, RewriterBase &rewriter,
                         ArrayRef<int32_t> order);

private:
  /// Initialize ops, opToId, and labels with default values.
  void initialize();
  llvm::SmallVector<Operation *> ops;
  llvm::DenseMap<Operation *, int64_t> opToId;
  llvm::SmallVector<int32_t> labels;
  Block *block = nullptr;
};
} // namespace aster
} // namespace mlir

#include "aster/Interfaces/SchedInterfaces.h.inc"

#endif // ASTER_INTERFACES_SCHEDINTERFACES_H
