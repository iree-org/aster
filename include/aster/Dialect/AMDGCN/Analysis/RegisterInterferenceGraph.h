//===- RegisterInterferenceGraph.h - Register interference graph -*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H

#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntEqClasses.h"
#include <optional>

namespace mlir {
class Operation;
class DataFlowSolver;
class SymbolTableCollection;
} // namespace mlir

namespace mlir::aster::amdgcn {
/// Register interference graph for AMDGCN register allocation. This analysis
/// uses the LivenessAnalysis to build an interference graph where
/// nodes represent allocations and edges connect allocations that overlap in
/// time.
struct RegisterInterferenceGraph : public Graph {
  enum class BuildMode {
    Minimal, /// Build the minimal graph to meet the liveness requirements.
    Full,    /// Add extra edges to ensure graph transformations are safe.
  };

  /// Create the interference graph for the given operation.
  /// This will load and run LivenessAnalysis internally.
  /// NOTE: Ranges will have consecutive node IDs, starting from 0.
  static FailureOr<RegisterInterferenceGraph>
  create(Operation *op, DataFlowSolver &solver,
         SymbolTableCollection &symbolTable,
         const RangeConstraintAnalysis &rangeAnalysis,
         BuildMode buildMode = BuildMode::Minimal);

  /// Print the interference graph.
  void print(raw_ostream &os) const;

  /// Get the node ID for a value. Returns -1 if not found.
  NodeID getNodeId(Value value) const;

  /// Get the value for a node ID.
  Value getValue(NodeID nodeId) const;

  /// Get all values in the graph.
  ArrayRef<Value> getValues() const { return values; }
  MutableArrayRef<Value> getValues() { return values; }

  /// Get the range information for a node ID. For any node returns the leader
  /// node ID and the range constraint.
  std::pair<NodeID, const RangeConstraint *> getRangeInfo(NodeID nodeId) const {
    if (nodeId >= static_cast<NodeID>(allocToRange.size()))
      return {nodeId, nullptr};
    NodeID rangeId = allocToRange[nodeId];
    return {rangeLeaders[rangeId], rangeAnalysis.getConstraintOrNull(rangeId)};
  }

private:
  RegisterInterferenceGraph(const RangeConstraintAnalysis &rangeAnalysis,
                            BuildMode buildMode)
      : Graph(/*directed=*/false), rangeAnalysis(rangeAnalysis),
        buildMode(buildMode) {}

  /// Run the interference analysis on the given operation.
  LogicalResult run(Operation *op, DataFlowSolver &solver);

  /// Handle a generic operation during graph construction.
  LogicalResult handleOp(Operation *op, DataFlowSolver &solver);

  /// Add edges between allocations.
  void addEdges(Value lhs, Value rhs);

  /// Add edges between all the related pairs of allocations in the given list.
  void addEdges(SmallVectorImpl<Value> &allocas);
  /// Add edges between the outs and the live values in the after set.
  void addEdges(SmallVectorImpl<Value> &outs, SmallVectorImpl<Value> &live);

  /// Get or create a node ID for an allocation.
  NodeID getOrCreateNodeId(Value allocation);

  /// The allocations in the graph.
  SmallVector<Value> values;
  /// Map from values to node IDs.
  llvm::DenseMap<Value, NodeID> valueToNodeId;
  /// Range constraint analysis.
  const RangeConstraintAnalysis &rangeAnalysis;
  /// Map from allocations in ranges to range IDs.
  SmallVector<NodeID> allocToRange;
  /// Leaders for register ranges (leader = start element of range).
  SmallVector<NodeID> rangeLeaders;
  /// Build mode.
  BuildMode buildMode;
};

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H
