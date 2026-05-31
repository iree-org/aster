//===- GraphColoring.cpp - Graph coloring for register allocation ---------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphColoring.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include <set>

#define DEBUG_TYPE "amdgcn-register-coloring"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

/// A physical register allocation: a contiguous range [begin, begin+size) of a
/// particular register kind.
struct Allocation {
  int16_t begin;
  int16_t size;
  RegisterKind kind;

  Allocation(int16_t begin, int16_t size, RegisterKind kind)
      : begin(begin), size(size), kind(kind) {}

  Allocation(AMDGCNRegisterTypeInterface regTy, int64_t numRegs)
      : begin(regTy.getAsRange().begin().getRegister()),
        size(static_cast<int16_t>(numRegs)), kind(regTy.getRegisterKind()) {}

  Register getBegin() const { return Register(begin); }
  Register getEnd() const { return Register(begin + size); }
  RegisterRange getRange() const { return RegisterRange(getBegin(), size, 1); }
  int16_t end() const { return begin + size; }

  bool operator<(const Allocation &other) const {
    return std::make_tuple(kind, begin) <
           std::make_tuple(other.kind, other.begin);
  }
};

//===----------------------------------------------------------------------===//
// AllocConstraints
//===----------------------------------------------------------------------===//

/// Physical register occupancy tracker. Finds the first free physical range
/// satisfying the given size and alignment.
///
/// numSGPR defaults to 102 — the fixed SGPR limit per wavefront on AMD GCN
/// architectures.
struct AllocConstraints {
  AllocConstraints(int32_t numVGPR = 256, int32_t numAGPR = 256,
                   int32_t numSGPR = 102)
      : numSGPR(numSGPR), numVGPR(numVGPR), numAGPR(numAGPR) {}

  /// Insert a given allocation.
  void insert(Allocation alloc);

  /// Allocate registers for a node. Returns failure if no suitable range could
  /// be found.
  FailureOr<Allocation> alloc(AMDGCNRegisterTypeInterface regTy,
                              int16_t numRegs, int16_t alignment);

  /// Clear all allocations.
  void clear();

  /// Print the allocation constraints.
  void print(raw_ostream &os) const;

private:
  /// The number of SGPRs per wavefront.
  const int32_t numSGPR;
  /// The number of VGPRs.
  const int32_t numVGPR;
  /// The number of AGPRs.
  const int32_t numAGPR;
  /// The allocation constraints. std::set is used here because the allocator
  /// requires ordered iteration by (kind, begin) to scan for gaps efficiently.
  std::set<Allocation> constraints;
};

//===----------------------------------------------------------------------===//
// GraphColoringImpl
//===----------------------------------------------------------------------===//

/// Pure graph coloring implementation. Holds no Value, no Op, no IRRewriter,
/// no RegisterInterferenceGraph, and no CoalescingInfo.
struct GraphColoringImpl {
  GraphColoringImpl(const aster::Graph &graph,
                    ArrayRef<NodeConstraint> nodeConstraints,
                    MutableArrayRef<AMDGCNRegisterTypeInterface> types,
                    Location loc, int32_t numVGPRs, int32_t numAGPRs,
                    int32_t numSGPRs)
      : graph(graph), nodeConstraints(nodeConstraints), types(types), loc(loc),
        allocConstraints(numVGPRs, numAGPRs, numSGPRs) {}

  /// Run graph coloring. Returns failure if any node cannot be colored.
  LogicalResult run();

private:
  /// Collect interference constraints from already-colored neighbors of nodeId.
  /// For range leaders, also collects from the neighbors of consecutive
  /// positions [nodeId+1, nodeId+numRegs).
  void collectConstraints(int32_t nodeId);

  /// Assign a physical register to nodeId (and all consecutive range positions
  /// if nodeConstraints[nodeId].numRegs > 1).
  LogicalResult colorNode(int32_t nodeId);

  const aster::Graph &graph;
  ArrayRef<NodeConstraint> nodeConstraints;
  MutableArrayRef<AMDGCNRegisterTypeInterface> types;
  Location loc;
  AllocConstraints allocConstraints;
  llvm::DenseSet<int32_t> visited;
};

} // namespace

//===----------------------------------------------------------------------===//
// AllocConstraints implementation
//===----------------------------------------------------------------------===//

void AllocConstraints::insert(Allocation alloc) { constraints.insert(alloc); }

FailureOr<Allocation> AllocConstraints::alloc(AMDGCNRegisterTypeInterface regTy,
                                              int16_t numRegs,
                                              int16_t alignment) {
  LDBG() << "  Allocating " << numRegs << " registers of kind "
         << regTy.getRegisterKind() << " with alignment " << alignment;

  int16_t maxRegs = 0;
  switch (regTy.getRegisterKind()) {
  case RegisterKind::SGPR:
    maxRegs = numSGPR;
    break;
  case RegisterKind::VGPR:
    maxRegs = numVGPR;
    break;
  case RegisterKind::AGPR:
    maxRegs = numAGPR;
    break;
  default:
    maxRegs = 1;
  }

  auto lb = constraints.lower_bound({0, 1, regTy.getRegisterKind()});
  auto ub = constraints.upper_bound({maxRegs, 1, regTy.getRegisterKind()});

  auto getStartAligned = [alignment](int64_t addr) {
    return static_cast<int16_t>(((addr + alignment - 1) / alignment) *
                                alignment);
  };

  int16_t start = 0;
  for (const Allocation &occupied : llvm::make_range(lb, ub)) {
    if (start + numRegs <= occupied.begin) {
      Allocation result = {start, numRegs, occupied.kind};
      constraints.insert(result);
      return result;
    }
    start = getStartAligned(occupied.end());
  }

  // Check if we can fit at the end.
  if (start + numRegs <= maxRegs) {
    Allocation result = {start, numRegs, regTy.getRegisterKind()};
    constraints.insert(result);
    return result;
  }

  return failure();
}

void AllocConstraints::clear() { constraints.clear(); }

void AllocConstraints::print(raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(constraints, os, [&](const Allocation &a) {
    os << a.getRange() << " : " << stringifyRegisterKind(a.kind);
  });
  os << "}";
}

//===----------------------------------------------------------------------===//
// GraphColoringImpl implementation
//===----------------------------------------------------------------------===//

void GraphColoringImpl::collectConstraints(int32_t nodeId) {
  LDBG() << " Collecting constraints for node[" << nodeId << "]";

  NodeConstraint nc = nodeConstraints[nodeId];
  int32_t numPositions = nc.numRegs;

  for (int32_t pos = 0; pos < numPositions; ++pos) {
    int32_t posId = nodeId + pos;
    for (auto [src, tgt] : graph.edges(posId)) {
      LDBG() << "  Inspecting neighbor[" << tgt << "]";
      AMDGCNRegisterTypeInterface regTy = types[tgt];
      if (!regTy.hasAllocatedSemantics())
        continue;
      assert(regTy.getAsRange().size() == 1 && "expected single register");
      allocConstraints.insert(Allocation(regTy, 1));
    }
  }
}

LogicalResult GraphColoringImpl::colorNode(int32_t nodeId) {
  NodeConstraint nc = nodeConstraints[nodeId];

  // Non-leader range positions are pre-marked visited by the leader's colorNode
  // loop and skipped by run() before colorNode is ever called on them.
  assert(nc.numRegs > 0 &&
         "colorNode must not be called on non-leader positions");

  AMDGCNRegisterTypeInterface regTy = types[nodeId];
  int16_t numRegs = static_cast<int16_t>(nc.numRegs);
  int16_t alignment = static_cast<int16_t>(nc.alignment);

  LDBG() << "Allocating node[" << nodeId << "] with " << numRegs
         << " regs, alignment " << alignment;

  FailureOr<Allocation> alloc =
      allocConstraints.alloc(regTy, numRegs, alignment);
  if (failed(alloc))
    return emitError(loc) << "failed to allocate the registers";

  // Write the allocated type for each consecutive range position.
  for (int32_t i = 0; i < numRegs; ++i) {
    Register reg = alloc->getBegin().getWithOffset(static_cast<int16_t>(i));
    types[nodeId + i] =
        cast<AMDGCNRegisterTypeInterface>(regTy.cloneRegisterType(reg));
    visited.insert(nodeId + i);
  }
  return success();
}

LogicalResult GraphColoringImpl::run() {
  for (auto [i, regTy] : llvm::enumerate(types)) {
    if (!visited.insert(static_cast<int32_t>(i)).second)
      continue;
    if (regTy.hasAllocatedSemantics())
      continue;
    allocConstraints.clear();
    collectConstraints(static_cast<int32_t>(i));
    if (failed(colorNode(static_cast<int32_t>(i))))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// colorGraph entry point
//===----------------------------------------------------------------------===//

LogicalResult mlir::aster::amdgcn::colorGraph(
    const aster::Graph &graph, ArrayRef<NodeConstraint> constraints,
    MutableArrayRef<AMDGCNRegisterTypeInterface> types, Location loc,
    int32_t numVGPRs, int32_t numAGPRs, int32_t numSGPRs) {
  return GraphColoringImpl(graph, constraints, types, loc, numVGPRs, numAGPRs,
                           numSGPRs)
      .run();
}
