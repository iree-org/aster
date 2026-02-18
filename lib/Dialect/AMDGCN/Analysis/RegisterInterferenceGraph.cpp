//===- RegisterInterferenceGraph.cpp - Register interference graph --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-register-interference"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

RegisterInterferenceGraph::NodeID
RegisterInterferenceGraph::getOrCreateNodeId(Value value) {
  auto it = valueToNodeId.find(value);
  if (it != valueToNodeId.end())
    return it->second;

  NodeID id = values.size();
  values.push_back(value);
  valueToNodeId[value] = id;
  return id;
}

RegisterInterferenceGraph::NodeID
RegisterInterferenceGraph::getNodeId(Value value) const {
  return valueToNodeId.lookup_or(value, -1);
}

Value RegisterInterferenceGraph::getValue(NodeID nodeId) const {
  if (nodeId < 0 || static_cast<size_t>(nodeId) >= values.size())
    return nullptr;
  return values[nodeId];
}

void RegisterInterferenceGraph::addEdges(Value lhs, Value rhs) {
  if (lhs == rhs)
    return;

  NodeID lhsId = getOrCreateNodeId(lhs);
  NodeID rhsId = getOrCreateNodeId(rhs);

  if (lhsId != rhsId)
    addEdge(lhsId, rhsId);

  LDBG_OS([&](raw_ostream &os) {
    os << "Added edge between values: \n";
    os << "  " << lhsId << ": ";
    lhs.printAsOperand(os, OpPrintingFlags());
    os << "\n  " << rhsId << ": ";
    rhs.printAsOperand(os, OpPrintingFlags());
  });
}

/// Helper function to sort and unique a list of values. This is used to
/// optimize
static void sortAndUniqueValues(SmallVectorImpl<Value> &values) {
  llvm::sort(values, [](Value a1, Value a2) {
    return std::make_tuple(a1.getType().getTypeID().getAsOpaquePointer(),
                           a1.getAsOpaquePointer()) <
           std::make_tuple(a2.getType().getTypeID().getAsOpaquePointer(),
                           a2.getAsOpaquePointer());
  });
  values.erase(llvm::unique(values), values.end());
}

void RegisterInterferenceGraph::addEdges(SmallVectorImpl<Value> &allocas) {
  if (allocas.empty())
    return;
  sortAndUniqueValues(allocas);
  ArrayRef<Value> allocasRef = allocas;
  // Add edges between all pairs of allocas.
  for (auto [i, a1] : llvm::enumerate(allocasRef)) {
    for (Value a2 : allocasRef.drop_front(i + 1)) {
      if (a1.getType().getTypeID() != a2.getType().getTypeID())
        break;
      addEdges(a1, a2);
    }
  }
}

void RegisterInterferenceGraph::addEdges(SmallVectorImpl<Value> &outs,
                                         SmallVectorImpl<Value> &live) {
  if (outs.empty() || live.empty())
    return;
  sortAndUniqueValues(outs);
  sortAndUniqueValues(live);
  ArrayRef<Value> outsRef = outs;
  ArrayRef<Value> liveRef = live;
  auto outIt = outsRef.begin();
  auto outEnd = outsRef.end();
  auto liveIt = liveRef.begin();
  auto liveEnd = liveRef.end();
  while (outIt != outEnd) {
    TypeID outKind = outIt->getType().getTypeID();
    // Find the end of the current kind in the outs.
    auto outOfKindEnd =
        llvm::find_if(llvm::make_range(outIt + 1, outEnd), [&](Value v) {
          return v.getType().getTypeID() != outKind;
        });

    // Find the first live value of the same kind.
    auto firstLiveOfKind =
        llvm::find_if(llvm::make_range(liveIt, liveEnd), [&](Value v) {
          return v.getType().getTypeID() == outKind;
        });

    // If there are no live values of the same kind, continue to the next kind.
    if (firstLiveOfKind == liveEnd) {
      outIt = outOfKindEnd;
      continue;
    }

    // Find the end of the live values of the same kind.
    auto liveOfKindEnd =
        llvm::find_if(llvm::make_range(firstLiveOfKind, liveEnd), [&](Value v) {
          return v.getType().getTypeID() != outKind;
        });

    // Add edges between all pairs of outs.
    for (Value a1 : llvm::make_range(outIt, outOfKindEnd)) {
      for (Value a2 : llvm::make_range(firstLiveOfKind, liveOfKindEnd)) {
        if (a1 == a2)
          continue;
        addEdges(a1, a2);
      }
    }

    // Move to the next kind.
    outIt = outOfKindEnd;
    liveIt = liveOfKindEnd;
  }
}

/// NOTE: We use the liveness set after the operation to build the interference
/// graph. The reason being that output registers must interfere with the set of
/// live registers after the instruction, as they will be written at the end of
/// the instruction's execution.
LogicalResult RegisterInterferenceGraph::handleOp(Operation *op,
                                                  DataFlowSolver &solver) {
  // Get the liveness state after the operation.
  const auto *state =
      solver.lookupState<LivenessState>(solver.getProgramPointAfter(op));
  const LivenessState::ValueSet *liveness =
      state ? state->getLiveValues() : nullptr;

  // Add the alloca to the graph.
  if (auto aOp = dyn_cast<AllocaOp>(op))
    getOrCreateNodeId(aOp.getResult());

  // If there's no liveness, return failure.
  if (!liveness)
    return op->emitError("found liveness with top state");

  LDBG_OS([&](raw_ostream &os) {
    os << "Liveness state after operation: "
       << OpWithFlags(op, OpPrintingFlags().skipRegions()) << "\n";
    os << "  ";
    state->print(os);
  });

  SmallVector<Value> allocas;

  // Add edges between the inputs of the RegInterferenceOp. Note that these
  // edges shouldn't be mixed with the edges between the live values, as the
  // semantics of the operation only establish interference between its inputs.
  if (auto regInterferenceOp = dyn_cast<RegInterferenceOp>(op)) {
    for (Value v : regInterferenceOp.getInputs()) {
      if (failed(getAllocasOrFailure(v, allocas)))
        return op->emitError("IR is not in the `unallocated` normal form");
    }
    addEdges(allocas);
    allocas.clear();
  }

  for (Value v : *liveness) {
    // Get the allocas in the liveness set.
    if (failed(getAllocasOrFailure(v, allocas)))
      return op->emitError("IR is not in the `unallocated` normal form");
  }

  SmallVector<Value, 5> outs;
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (Value out : instOp.getInstOuts()) {
      if (failed(getAllocasOrFailure(out, outs)))
        return op->emitError("IR is not in the `unallocated` normal form");
    }
    // Force all outs to interfere with each other to avoid reuse of the same
    // register in the outs of the same operation.
    addEdges(outs);
  }

  // Add edges between the outputs and the live values in the after set.
  addEdges(outs, allocas);
  return success();
}

LogicalResult RegisterInterferenceGraph::run(Operation *op,
                                             DataFlowSolver &solver) {
  LDBG() << "Running register interference analysis on operation: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  // Walk the operation tree to build the interference graph.
  WalkResult result = op->walk([&](Operation *wOp) {
    if (op == wOp)
      return WalkResult::advance();
    if (failed(handleOp(wOp, solver)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Check if the walk was interrupted.
  if (result.wasInterrupted())
    return failure();

  // Set the number of nodes and compress the graph.
  setNumNodes(values.size());
  compress();
  LDBG_OS([&](raw_ostream &os) {
    os << "Register interference graph:\n";
    print(os);
  });
  return success();
}

FailureOr<RegisterInterferenceGraph>
RegisterInterferenceGraph::create(Operation *op, DataFlowSolver &solver,
                                  SymbolTableCollection &symbolTable) {
  // Load the register liveness analysis.
  solver.load<LivenessAnalysis>(symbolTable);
  mlir::dataflow::loadBaselineAnalyses(solver);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op))) {
    LDBG() << "failed to run register liveness analysis";
    return failure();
  }

  // Build the graph.
  RegisterInterferenceGraph graph;
  if (failed(graph.run(op, solver)))
    return failure();

  return graph;
}

void RegisterInterferenceGraph::print(raw_ostream &os) const {
  assert(isCompressed() && "Graph must be compressed before printing");
  os << "graph RegisterInterference {\n";
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
