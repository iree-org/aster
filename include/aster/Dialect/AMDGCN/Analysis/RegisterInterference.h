//===- RegisterInterference.h - Register interference analysis ---*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCE_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCE_H

#include "aster/Dialect/AMDGCN/Analysis/RegisterLiveness.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
class Operation;
class DataFlowSolver;
class SymbolTableCollection;
} // namespace mlir

namespace mlir::aster::amdgcn {

/// Register interference graph analysis for AMDGCN register allocation.
/// This analysis uses the RegisterLiveness analysis to build an interference
/// graph where nodes represent values and edges connect values that are
/// simultaneously live at some program point.
struct RegisterInterference : public Graph {
  /// Create an interference graph for the given operation.
  /// This will load and run RegisterLiveness internally.
  static FailureOr<RegisterInterference>
  create(Operation *op, DataFlowSolver &solver,
         SymbolTableCollection &symbolTable);

  /// Print the interference graph.
  void print(raw_ostream &os) const;

  /// Get the node ID for a value. Returns -1 if not found.
  NodeID getNodeId(Value value) const;

  /// Get the value for a node ID.
  Value getValue(NodeID nodeId) const;

  /// Get all values in the graph.
  ArrayRef<Value> getValues() const { return values; }
  MutableArrayRef<Value> getValues() { return values; }

private:
  RegisterInterference(DataFlowSolver &solver)
      : Graph(/*directed=*/false), solver(solver) {}

  /// Run the interference analysis on the given operation.
  LogicalResult run(Operation *op);

  /// Handle a generic operation during graph construction.
  LogicalResult handleOp(Operation *op);

  /// Add edges between values.
  void addEdges(Value lhs, Value rhs);

  /// Get or create a node ID for a value.
  NodeID getOrCreateNodeId(Value value);

  DataFlowSolver &solver;
  SmallVector<Value> values;
  llvm::DenseMap<Value, NodeID> valueToNodeId;
  // Scratch space to avoid repeated allocations.
  SmallVector<Value> liveValuesScratch;
};

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCE_H
