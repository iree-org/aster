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

#include "aster/Analysis/DPSAliasAnalysis.h"
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
class DPSAliasAnalysis;
/// Interference graph analysis.
struct InterferenceAnalysis : public Graph {
  /// Create an interference graph for the given operation and data flow solver.
  static FailureOr<InterferenceAnalysis>
  create(Operation *op, DataFlowSolver &solver, DPSAliasAnalysis *aliasAnalysis);
  static FailureOr<InterferenceAnalysis>
  create(Operation *op, DataFlowSolver &solver,
         SymbolTableCollection &symbolTable);

  /// Print the interference graph.
  void print(raw_ostream &os) const;

  /// Get the equivalence class IDs associated with a value.
  llvm::ArrayRef<EqClassID> getEqClassIds(Value value) const;

  /// Get the underlying DPS alias analysis.
  const DPSAliasAnalysis *getAnalysis() const { return aliasAnalysis; }
  const DPSAliasAnalysis *operator->() const { return aliasAnalysis; }

private:
  InterferenceAnalysis(DataFlowSolver &solver, DPSAliasAnalysis *aliasAnalysis)
      : Graph(false), solver(solver), aliasAnalysis(aliasAnalysis) {}
  /// Handle a generic operation during graph construction.
  LogicalResult handleOp(Operation *op);
  /// Add edges between equivalence classes.
  void addEdges(Value lhsV, Value rhsV, llvm::ArrayRef<int32_t> lhs,
                llvm::ArrayRef<int32_t> rhs);
  DataFlowSolver &solver;
  DPSAliasAnalysis *aliasAnalysis;
  // Scratch space to avoid repeated allocations.
  llvm::SmallVector<std::pair<ResourceTypeInterface, Value>> liveRegsScratch;
  SmallVector<llvm::ArrayRef<EqClassID>> eqClassIdsScratch;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_INTERFERENCEANALYSIS_H
