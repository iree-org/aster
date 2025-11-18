//===- RangeAnalysis.cpp - Range analysis --------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/RangeAnalysis.h"
#include "aster/Analysis/VariableAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/ResourceInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include <cstddef>
#include <iterator>

#define DEBUG_TYPE "range-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct RangeAnalysisImpl {
  using RangeList = SmallVector<Range *>;
  RangeAnalysisImpl(const VariableAnalysis *analysis,
                    SmallVector<Range> &ranges,
                    SmallVector<RangeAllocation> &allocations, Graph &graph)
      : analysis(analysis), ranges(ranges), allocations(allocations),
        graph(graph) {}

  FailureOr<DenseMap<VariableID, int32_t>> computeAnalysis(Operation *op);

private:
  void createGraph();
  FailureOr<DenseMap<VariableID, int32_t>> computeSatisfiability(Location loc);
  const VariableAnalysis *analysis;
  SmallVector<Range> &ranges;
  SmallVector<RangeAllocation> &allocations;
  Graph &graph;
};
} // namespace

/// Compare two ranges for sorting.
static bool cmpRanges(const Range &lhs, const Range &rhs) {
  auto getTuple = [](const Range &r) {
    auto ty = static_cast<ResourceTypeInterface>(r.getRegisterType());
    return std::make_tuple(ty, r.getVariableIds(), r.getOp().getOperation());
  };
  return getTuple(lhs) < getTuple(rhs);
}

/// Add an edge between two ranges in the graph.
static void addEdge(Graph &graph, const Range &src, const Range &tgt,
                    size_t srcId, size_t tgtId) {
  ArrayRef<VariableID> srcIds = src.getVariableIds();
  ArrayRef<VariableID> tgtIds = tgt.getVariableIds();
  assert(!srcIds.empty() && !tgtIds.empty());

  // Find the largest common subsequence between srcIds and tgtIds
  size_t srcStart = 0, tgtStart = 0;
  size_t commonLength = 0;
  for (size_t i = 0; i < srcIds.size() && commonLength <= 0; ++i) {
    for (size_t j = 0; j < tgtIds.size() && commonLength <= 0; ++j) {
      size_t si = i, ti = j;
      srcStart = i;
      tgtStart = j;
      while (si < srcIds.size() && ti < tgtIds.size() &&
             srcIds[si] == tgtIds[ti]) {
        commonLength++;
        si++;
        ti++;
      }
    }
  }

  // The ranges are unrelated.
  if (commonLength == 0)
    return;

  // There's a contiguous overlap at the beginning of both sequences, add an
  // edge from largest sequence to smallest.
  size_t smallestSize =
      srcIds.size() <= tgtIds.size() ? srcIds.size() : tgtIds.size();
  if (srcStart == 0 && tgtStart == 0 && commonLength == smallestSize) {
    size_t lId = srcIds.size() >= tgtIds.size() ? srcId : tgtId;
    size_t sId = srcIds.size() >= tgtIds.size() ? tgtId : srcId;
    graph.addEdge(lId, sId);
    return;
  }

  // There's a contiguous overlap at the end of tgtIds and the beginning of
  // srcIds.
  if (srcStart == 0 && tgtStart + commonLength == tgtIds.size()) {
    graph.addEdge(tgtId, srcId);
    return;
  }

  // There's a contiguous overlap at the end of srcIds and the beginning of
  // tgtIds.
  if (tgtStart == 0 && srcStart + commonLength == srcIds.size()) {
    graph.addEdge(srcId, tgtId);
    return;
  }

  // The ranges have common elements but neither is a partial subrange of the
  // other.
  graph.addEdge(srcId, tgtId);
  graph.addEdge(tgtId, srcId);
}

void RangeAnalysisImpl::createGraph() {
  for (size_t i = 0; i < ranges.size(); ++i) {
    for (size_t j = i + 1; j < ranges.size(); ++j) {
      if (ranges[i].getRegisterType().getResource() !=
          ranges[j].getRegisterType().getResource())
        break;
      addEdge(graph, ranges[i], ranges[j], i, j);
    }
  }
  graph.setNumNodes(ranges.size());
  graph.compress();
}

LogicalResult RangeAllocation::setAlignment(Location loc, VariableID id,
                                            int32_t alignCtr,
                                            MakeRegisterRangeOp owner) {
  assert(alignCtr > 0 && "Alignment must be positive");
  ArrayRef<VariableID> allocatedVars = allocatedVariables.getArrayRef();
  ptrdiff_t pos =
      std::distance(allocatedVars.begin(), llvm::find(allocatedVars, id));
  for (int32_t a = 1; a <= alignCtr; ++a) {
    if ((pos + alignment * a) % alignCtr == 0) {
      alignment = a * alignment;
      return success();
    }
  }
  emitError(loc) << "Unsatisfiable alignment constraint from: " << Value(owner);
  return failure();
}

FailureOr<DenseMap<VariableID, int32_t>>
RangeAnalysisImpl::computeSatisfiability(Location loc) {
  using NodeID = Graph::NodeID;
  FailureOr<SmallVector<NodeID>> graphSort = graph.topologicalSort();
  if (failed(graphSort)) {
    emitError(loc) << "There are un-allocatable ranges";
    return failure();
  }
  LDBG() << "Topological sort result: " << llvm::interleaved_array(*graphSort);
  DenseMap<VariableID, int32_t> allocMap;
  auto setAlloc = [&](const Range &range, int32_t alloc) {
    RangeAllocation &allocation = allocations[alloc];
    for (VariableID var : range.getVariableIds()) {
      allocMap[var] = alloc;
      allocation.pushVariable(var);
    }
    return allocation.setAlignment(
        analysis->getVariables()[allocation.startVariable()].getLoc(),
        range.startVariable(), range.getRegisterType().getAsRange().alignment(),
        range.getOp());
  };
  SmallVector<int> visits(ranges.size(), 0);
  for (NodeID node : *graphSort) {
    Range range = ranges[node];
    if (visits[node] == 0) {
      allocations.push_back(RangeAllocation(range.startVariable()));
      if (failed(setAlloc(range, allocations.size() - 1)))
        return failure();
    }
    if (visits[node] > 1)
      continue;
    ++visits[node];
    int32_t alloc = allocMap.lookup_or(range.startVariable(), -1);
    assert(alloc != -1 && "invalid alloc");
    for (auto [src, tgt] : graph.edges(node)) {
      if (failed(setAlloc(ranges[tgt], alloc)))
        return failure();
      ++visits[tgt];
    }
  }
  return allocMap;
}

FailureOr<DenseMap<VariableID, int32_t>>
RangeAnalysisImpl::computeAnalysis(Operation *op) {
  op->walk([&](MakeRegisterRangeOp range) {
    ranges.push_back(Range(range, analysis->getVariableIds(range)));
  });
  llvm::sort(ranges, cmpRanges);
  LDBG_OS([&](raw_ostream &os) {
    os << "Sorted ranges:\n";
    llvm::interleave(
        llvm::enumerate(ranges), os,
        [&](auto it) {
          const Range &range = it.value();
          os << "  " << it.index() << ": "
             << llvm::interleaved_array(range.getVariableIds())
             << ", op: " << range.getOp();
        },
        "\n");
  });
  createGraph();
  LDBG_OS([&](raw_ostream &os) {
    os << "Graph: \n";
    graph.print(os);
  });
  return computeSatisfiability(op->getLoc());
}

RangeAnalysis RangeAnalysis::create(Operation *topOp,
                                    const VariableAnalysis *analysis) {
  RangeAnalysis ra(analysis);
  RangeAnalysisImpl impl(analysis, ra.ranges, ra.allocations, ra.graph);
  ra.allocationMap = impl.computeAnalysis(topOp);
  LDBG_OS([&](raw_ostream &os) {
    if (failed(ra.allocationMap)) {
      os << "Range constraints are not satisfiable";
      return;
    }
    os << "Range constraints mapping: ";
    llvm::interleaveComma(*ra.allocationMap, os, [&](const auto &it) {
      os << "(" << it.first << "->"
         << ra.allocations[it.second].getAllocatedVariables().front() << ")";
    });
    os << "\nRange allocations:\n";
    llvm::interleave(
        ra.allocations, os,
        [&](const RangeAllocation &alloc) {
          os << llvm::interleaved_array(alloc.getAllocatedVariables());
        },
        "\n");
  });
  return ra;
}
