//===- LDSInterferenceGraph.cpp - LDS Interference Graph ------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/LDSInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/Analysis/BufferAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "lds-interference-graph"

using namespace mlir;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Check for conflicting liveness states (top) and emit errors.
static LogicalResult checkTopStates(Operation *operation,
                                    const BufferState *afterState) {
  for (auto [buffer, state] : afterState->getBuffers()) {
    if (state == BufferState::State::Top)
      return operation->emitError() << "LDS buffer has top state";
  }
  return success();
}

/// Check for get_lds_offset operations on dead or invalid buffers.
static LogicalResult checkGetLDSOffset(GetLDSOffsetOp getLDSOffsetOp,
                                       DataFlowSolver &solver) {
  Value buffer = getLDSOffsetOp.getBuffer();
  const auto *beforeState = solver.lookupState<BufferState>(
      solver.getProgramPointBefore(getLDSOffsetOp));
  if (!beforeState) {
    return getLDSOffsetOp.emitError()
           << "missing buffer state before get_lds_offset";
  }

  BufferState::State state = beforeState->getBufferState(buffer);
  if (state == BufferState::State::Dead) {
    return getLDSOffsetOp.emitError()
           << "get_lds_offset operates on a dead buffer";
  }

  for (Operation *user : getLDSOffsetOp.getResult().getUsers()) {
    const auto *beforeState =
        solver.lookupState<BufferState>(solver.getProgramPointBefore(user));
    if (!beforeState)
      return user->emitError() << "missing buffer state before offset use";
    BufferState::State state = beforeState->getBufferState(buffer);
    if (state != BufferState::State::Live) {
      return getLDSOffsetOp.emitError()
             << "get_lds_offset use operates on a non-live buffer";
    }
  }
  return success();
}

/// Add interference edges between all pairs of live buffers.
static void
addInterferenceEdges(ArrayRef<LDSInterferenceGraph::NodeId> liveNodes,
                     LDSInterferenceGraph &graph) {
  for (int i = 0, end = static_cast<int>(liveNodes.size()); i < end; ++i) {
    for (int j = i + 1; j < end; ++j) {
      if (liveNodes[i] == liveNodes[j])
        continue;
      graph.addEdge(liveNodes[i], liveNodes[j]);
      LDBG() << "Added interference edge between nodes: " << liveNodes[i]
             << " and " << liveNodes[j];
    }
  }
}

//===----------------------------------------------------------------------===//
// LDSInterferenceGraph
//===----------------------------------------------------------------------===//

LDSInterferenceGraph::NodeId LDSInterferenceGraph::addNode(AllocLDSOp allocOp,
                                                           int64_t size,
                                                           int64_t alignment) {
  NodeId id = allocNodes.size();
  allocNodes.emplace_back(allocOp, size, alignment);
  allocToNodeId[allocOp.getBuffer()] = id;
  return id;
}

LDSInterferenceGraph::NodeId
LDSInterferenceGraph::getNodeId(Value buffer) const {
  return allocToNodeId.lookup_or(buffer, -1);
}

LogicalResult LDSInterferenceGraph::buildAndVerify(Operation *op,
                                                   DataFlowSolver &solver) {
  WalkResult walkResult =
      op->walk<WalkOrder::PreOrder>([&](Operation *operation) -> WalkResult {
        // Collect AllocLDSOp nodes.
        if (auto allocOp = dyn_cast<AllocLDSOp>(operation)) {
          OpFoldResult sizeResult = allocOp.getSize();
          auto constSize = getConstantIntValue(sizeResult);
          if (!constSize) {
            allocOp.emitError() << "LDS allocation must have a constant size";
            return WalkResult::interrupt();
          }

          int64_t alignment = allocOp.getAlignment();
          int64_t id = addNode(allocOp, *constSize, alignment);
          LDBG() << "Added AllocLDSOp node for buffer: " << allocOp.getBuffer()
                 << " (node ID: " << id << ")";
          (void)id;
        }

        // Early advance for function ops.
        if (isa<FunctionOpInterface>(operation))
          return WalkResult::advance();

        // Get the buffer state after this operation.
        auto *afterState = solver.lookupState<BufferState>(
            solver.getProgramPointAfter(operation));
        if (!afterState) {
          operation->emitError() << "missing buffer state after operation";
          return WalkResult::interrupt();
        }

        // Check for conflicting liveness states.
        if (failed(checkTopStates(operation, afterState)))
          return WalkResult::interrupt();

        // Check get_lds_offset on dead buffers.
        if (auto getLDSOffsetOp = dyn_cast<GetLDSOffsetOp>(operation)) {
          if (failed(checkGetLDSOffset(getLDSOffsetOp, solver)))
            return WalkResult::interrupt();
        }

        LDBG() << "Collecting live buffers after operation: "
               << OpWithFlags(operation, OpPrintingFlags().skipRegions())
               << "\n  with state: " << *afterState;
        // Collect live buffers and add interference edges.
        SmallVector<LDSInterferenceGraph::NodeId> liveNodes;
        for (auto [buffer, state] : afterState->getBuffers()) {
          if (state == BufferState::State::Live) {
            if (LDSInterferenceGraph::NodeId nodeId = getNodeId(buffer);
                nodeId >= 0)
              liveNodes.push_back(nodeId);
          }
        }
        addInterferenceEdges(liveNodes, *this);
        return WalkResult::advance();
      });

  if (walkResult.wasInterrupted())
    return failure();

  setNumNodes(getAllocNodes().size());
  compress();
  return success();
}

FailureOr<LDSInterferenceGraph>
LDSInterferenceGraph::create(Operation *op, DominanceInfo &domInfo) {
  // Run buffer analysis.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<BufferAnalysis>(domInfo);

  if (failed(solver.initializeAndRun(op)))
    return op->emitError() << "failed to run buffer analysis";

  // Build the graph.
  LDSInterferenceGraph graph;
  if (failed(graph.buildAndVerify(op, solver)))
    return failure();

  return graph;
}

void LDSInterferenceGraph::print(raw_ostream &os) const {
  assert(isCompressed() && "Graph must be compressed before printing");
  os << "graph LDSInterferenceGraph {\n";
  llvm::interleave(
      nodes(), os,
      [&](NodeID node) {
        const LDSAllocNode &allocNode = allocNodes[node];
        AllocLDSOp allocOp = allocNode.allocOp;
        os << "  " << node << " [label=\"" << node << ": ";
        allocOp.getBuffer().printAsOperand(os, OpPrintingFlags());
        os << " (size=" << allocNode.size << ", align=" << allocNode.alignment
           << ")\"];";
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
