//===- InterferenceAnalysis.cpp - Interference graph analysis -------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/InterferenceAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Analysis/VariableAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "interference-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

llvm::ArrayRef<VariableID>
InterferenceAnalysis::getVariableIds(Value value) const {
  const auto *lattice = solver.lookupState<dataflow::Lattice<Variable>>(value);
  assert(lattice && "missing variable lattice");
  const Variable &variable = lattice->getValue();
  assert(!(variable.isTop() || variable.isUninitialized()) &&
         "invalid variable value");
  return variable.getVariableIds();
}

void InterferenceAnalysis::addEdges(Value lhsV, Value rhsV,
                                    llvm::ArrayRef<VariableID> lhs,
                                    llvm::ArrayRef<VariableID> rhs) {
  // Add edges between all pairs of variables in lhs and rhs.
  for (VariableID l : lhs) {
    for (VariableID r : rhs) {
      if (l != r)
        addEdge(l, r);
    }
  }
  LDBG_OS([&](raw_ostream &os) {
    os << "Added edge between variables: \n";
    os << "  " << llvm::interleaved_array(lhs) << ": " << lhsV;
    os << "  " << llvm::interleaved_array(rhs) << ": " << rhsV;
  });
}

LogicalResult InterferenceAnalysis::handleOp(Operation *op) {
  // Skip MakeRegisterRangeOp and SplitRegisterRangeOp as they don't affect
  // interference.
  if (isa<MakeRegisterRangeOp, SplitRegisterRangeOp>(op))
    return success();

  // Get the liveness lattice before the operation.
  const auto *lattice =
      solver.lookupState<LivenessState>(solver.getProgramPointBefore(op));
  assert(lattice && "missing liveness lattice");
  if (lattice->isTop())
    return op->emitError() << "liveness lattice is top";
  const LivenessState::LiveSet &liveValues = *lattice->getLiveValues();

  // Collect live registers.
  llvm::SmallVector<std::pair<RegisterTypeInterface, Value>> liveRegs;
  for (Value v : liveValues) {
    auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
    if (!regTy)
      continue;
    liveRegs.push_back({regTy, v});
  }
  for (Value v : op->getResults()) {
    auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
    if (!regTy)
      continue;
    liveRegs.push_back({regTy, v});
  }

  // Skip if there are no live registers.
  if (liveRegs.empty())
    return success();
  LDBG() << "Computing interference for: `"
         << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << "` with live values: ";
  LDBG_OS([&](raw_ostream &os) {
    os << "  " << llvm::interleaved(llvm::make_second_range(liveRegs));
  });

  // Sort live registers by type and value to group similar registers together.
  llvm::sort(liveRegs, [](const std::pair<RegisterTypeInterface, Value> &lhs,
                          const std::pair<RegisterTypeInterface, Value> &rhs) {
    return std::make_tuple(lhs.first.getRegisterKindAsInt(),
                           lhs.first.getRegisterKind().getAsOpaquePointer(),
                           lhs.first.getAsOpaquePointer(),
                           lhs.second.getAsOpaquePointer()) <
           std::make_tuple(rhs.first.getRegisterKindAsInt(),
                           rhs.first.getRegisterKind().getAsOpaquePointer(),
                           rhs.first.getAsOpaquePointer(),
                           rhs.second.getAsOpaquePointer());
  });

  // Add edges between all pairs of variables in lhs and rhs.
  for (int i = 0, end = static_cast<int>(liveRegs.size()); i < end; ++i) {
    for (int j = i + 1; j < end; ++j) {
      // Skip if the register kinds differ.
      if (liveRegs[i].first.getRegisterKind() !=
          liveRegs[j].first.getRegisterKind())
        break;
      llvm::ArrayRef<VariableID> u = getVariableIds(liveRegs[i].second);
      llvm::ArrayRef<VariableID> v = getVariableIds(liveRegs[j].second);
      addEdges(liveRegs[i].second, liveRegs[j].second, u, v);
    }
  }
  return success();
}

FailureOr<InterferenceAnalysis>
InterferenceAnalysis::create(Operation *op, DataFlowSolver &solver,
                             VariableAnalysis *varAnalysis) {
  // Check for ill-formed IR.
  if (varAnalysis->isIllFormedIR())
    return op->emitError() << "ill-formed IR detected";

  InterferenceAnalysis graph(solver, varAnalysis);
  // Walk the operation tree to build the interference graph.
  WalkResult result = op->walk([&](Operation *wOp) {
    if (op == wOp)
      return WalkResult::advance();
    if (failed(graph.handleOp(wOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Check if the walk was interrupted.
  if (result.wasInterrupted())
    return failure();

  // Set the number of nodes and compress the graph.
  graph.setNumNodes(varAnalysis->getVariables().size());
  graph.compress();
  LDBG_OS([&](raw_ostream &os) {
    os << "Interference graph:\n";
    graph.print(os);
  });
  return graph;
}

FailureOr<InterferenceAnalysis>
InterferenceAnalysis::create(Operation *op, DataFlowSolver &solver,
                             SymbolTableCollection &symbolTable) {
  // Load the necessary analyses.
  dataflow::loadBaselineAnalyses(solver);
  auto *varAnalysis = solver.load<aster::VariableAnalysis>();
  solver.load<aster::LivenessAnalysis>(symbolTable);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op)))
    return failure();
  return create(op, solver, varAnalysis);
}

void InterferenceAnalysis::print(raw_ostream &os) const {
  assert(compressed && "Graph must be compressed before printing");
  ArrayRef<Value> variables = varAnalysis->getVariables();
  os << "graph InterferenceAnalysis {\n";
  llvm::interleave(
      nodes(), os,
      [&](NodeID node) {
        os << "  " << node << " [label=\"" << node << ", ";
        variables[node].printAsOperand(os, OpPrintingFlags());
        os << "\"];";
      },
      "\n");
  os << "\n";
  for (const Edge &edge : edges()) {
    NodeID src = edge.first;
    NodeID tgt = edge.second;
    if (src > tgt)
      continue;
    os << "  " << src << " -- " << tgt << ";\n";
  }
  os << "}";
}
