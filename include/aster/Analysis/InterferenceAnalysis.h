//===- InterferenceAnalysis.h - Interference analysis ------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_INTERFERENCEANALYSIS_H
#define ASTER_ANALYSIS_INTERFERENCEANALYSIS_H

#include "aster/Analysis/VariableAnalysis.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mlir {
class Operation;
class DataFlowSolver;
class SymbolTableCollection;
} // namespace mlir

namespace mlir::aster {
class VariableAnalysis;
/// Interference graph analysis.
struct InterferenceAnalysis : public Graph {
  /// Create an interference graph for the given operation and data flow solver.
  static FailureOr<InterferenceAnalysis>
  create(Operation *op, DataFlowSolver &solver, VariableAnalysis *varAnalysis);
  static FailureOr<InterferenceAnalysis>
  create(Operation *op, DataFlowSolver &solver,
         SymbolTableCollection &symbolTable);

  /// Print the interference graph.
  void print(raw_ostream &os) const;

  /// Get the variable IDs associated with a value.
  llvm::ArrayRef<VariableID> getVariableIds(Value value) const;

  /// Get the underlying variable analysis.
  const VariableAnalysis *getAnalysis() const { return varAnalysis; }
  const VariableAnalysis *operator->() const { return varAnalysis; }

private:
  InterferenceAnalysis(DataFlowSolver &solver, VariableAnalysis *varAnalysis)
      : Graph(false), solver(solver), varAnalysis(varAnalysis) {}
  /// Handle a generic operation during graph construction.
  LogicalResult handleOp(Operation *op);
  /// Add edges between variables.
  void addEdges(Value lhsV, Value rhsV, llvm::ArrayRef<int32_t> lhs,
                llvm::ArrayRef<int32_t> rhs);
  DataFlowSolver &solver;
  VariableAnalysis *varAnalysis;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_INTERFERENCEANALYSIS_H
