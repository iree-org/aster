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
#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/ResourceInterfaces.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <utility>

#define DEBUG_TYPE "interference-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

llvm::ArrayRef<EqClassID>
InterferenceAnalysis::getEqClassIds(Value value) const {
  const auto *lattice = solver.lookupState<dataflow::Lattice<AliasEquivalenceClass>>(value);
  assert(lattice && "missing equivalence class lattice");
  const AliasEquivalenceClass &eqClass = lattice->getValue();
  assert(!(eqClass.isTop() || eqClass.isUninitialized()) &&
         "invalid equivalence class value");
  return eqClass.getEqClassIds();
}

void InterferenceAnalysis::addEdges(Value lhsV, Value rhsV,
                                    llvm::ArrayRef<EqClassID> lhs,
                                    llvm::ArrayRef<EqClassID> rhs) {
  // Add edges between all pairs of equivalence classes in lhs and rhs.
  for (EqClassID l : lhs) {
    for (EqClassID r : rhs) {
      if (l != r)
        addEdge(l, r);
    }
  }
  LDBG_OS([&](raw_ostream &os) {
    os << "Added edge between equivalence classes: \n";
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
  llvm::SmallVectorImpl<std::pair<ResourceTypeInterface, Value>> &liveRegs =
      liveRegsScratch;
  liveRegs.clear();
  for (Value v : liveValues) {
    auto regTy = dyn_cast<ResourceTypeInterface>(v.getType());
    if (!regTy)
      continue;
    liveRegs.push_back({regTy, v});
  }
  for (Value v : op->getResults()) {
    auto regTy = dyn_cast<ResourceTypeInterface>(v.getType());
    if (!regTy)
      continue;
    liveRegs.push_back({regTy, v});
  }
  if (auto iOp = dyn_cast<amdgcn::RegInterferenceOp>(op)) {
    for (Value v : iOp.getInputs())
      liveRegs.push_back({cast<ResourceTypeInterface>(v.getType()), v});
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
  llvm::sort(liveRegs, [](const std::pair<ResourceTypeInterface, Value> &lhs,
                          const std::pair<ResourceTypeInterface, Value> &rhs) {
    return std::make_tuple(lhs.first, lhs.second.getAsOpaquePointer()) <
           std::make_tuple(rhs.first, rhs.second.getAsOpaquePointer());
  });

  SmallVectorImpl<llvm::ArrayRef<EqClassID>> &eqClassIds = eqClassIdsScratch;
  eqClassIds.clear();
  for (const auto &[rTy, value] : liveRegs)
    eqClassIds.push_back(getEqClassIds(value));

  // Add edges between all pairs of equivalence classes in lhs and rhs.
  for (int i = 0, end = static_cast<int>(liveRegs.size()); i < end; ++i) {
    for (int j = i + 1; j < end; ++j) {
      // Skip if the register kinds differ.
      if (liveRegs[i].first.getResource() != liveRegs[j].first.getResource())
        break;
      llvm::ArrayRef<EqClassID> u = eqClassIds[i];
      llvm::ArrayRef<EqClassID> v = eqClassIds[j];
      addEdges(liveRegs[i].second, liveRegs[j].second, u, v);
    }
  }
  return success();
}

FailureOr<InterferenceAnalysis>
InterferenceAnalysis::create(Operation *op, DataFlowSolver &solver,
                             DPSAliasAnalysis *aliasAnalysis) {
  // Check for ill-formed IR.
  if (aliasAnalysis->isIllFormedIR())
    return op->emitError() << "ill-formed IR detected";

  InterferenceAnalysis graph(solver, aliasAnalysis);
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
  graph.setNumNodes(aliasAnalysis->getValues().size());
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
  auto *aliasAnalysis = solver.load<aster::DPSAliasAnalysis>();
  solver.load<aster::LivenessAnalysis>(symbolTable);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op)))
    return failure();
  return create(op, solver, aliasAnalysis);
}

void InterferenceAnalysis::print(raw_ostream &os) const {
  assert(compressed && "Graph must be compressed before printing");
  ArrayRef<Value> values = aliasAnalysis->getValues();
  os << "graph InterferenceAnalysis {\n";
  llvm::interleave(
      nodes(), os,
      [&](NodeID node) {
        os << "  " << node << " [label=\"" << node << ", ";
        values[node].printAsOperand(os, OpPrintingFlags());
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
