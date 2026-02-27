//===- LDSAllocPass.cpp - LDS Buffer Allocation Pass ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LDS buffer allocation analysis pass.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/LDSInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include <cstdint>
#include <set>

#define DEBUG_TYPE "amdgcn-lds-alloc"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LDSALLOC
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// LDSAllocator
//===----------------------------------------------------------------------===//
/// A memory allocation.
struct Allocation {
  int64_t begin;
  int64_t end;

  bool operator<(const Allocation &other) const { return begin < other.begin; }
};

/// The allocation constraints.
struct AllocConstraints {
  static constexpr int64_t kMaxMemory = (1 << 16); // 64KB

  /// Insert a given allocation, returns failure if it overlaps with existing
  /// ones.
  LogicalResult insert(Allocation alloc);

  /// Allocate memory for a given node, returns failure if no suitable region
  /// could be found.
  FailureOr<Allocation> alloc(LDSAllocNode alloc, int64_t startPos);

  /// Clear all allocations.
  void clear();

  /// Print the allocation constraints.
  void print(raw_ostream &os) const;

private:
  std::set<Allocation> allocations;
};

/// A greedy allocator for LDS memory based on the interference graph.
/// The allocator traverses nodes in breadth-first order and assigns offsets
/// to buffers.
struct LDSAllocator {
  using NodeId = LDSInterferenceGraph::NodeId;
  LDSAllocator(const LDSInterferenceGraph &graph) : graph(graph) {}

  /// Get the total LDS memory used.
  int64_t getTotalSize() const { return totalSize; }

  /// Run the allocator on all nodes in BFS order, returns failure if an
  /// allocation request cannot be satisfied.
  LogicalResult run(aster::GPUFuncInterface op);

private:
  /// Collect the allocation constraints for the given node.
  LogicalResult collectConstraints(NodeId nodeId, ArrayRef<LDSAllocNode> nodes);
  /// Allocate memory for a buffer node.
  LogicalResult alloc(LDSAllocNode node);

  const LDSInterferenceGraph &graph;
  AllocConstraints constraints;
  int64_t startPos = 0;
  int64_t totalSize = 0;
};

//===----------------------------------------------------------------------===//
// LDSAllocPass
//===----------------------------------------------------------------------===//

struct LDSAllocPass
    : public mlir::aster::amdgcn::impl::LDSAllocBase<LDSAllocPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();

    WalkResult walkResult =
        op->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          auto kernOp = dyn_cast<aster::GPUFuncInterface>(op);
          if (!kernOp)
            return WalkResult::advance();

          FailureOr<LDSInterferenceGraph> graph =
              LDSInterferenceGraph::create(kernOp, domInfo);
          if (failed(graph))
            return WalkResult::interrupt();

          LDSAllocator allocator(*graph);
          if (failed(allocator.run(kernOp)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// AllocConstraints
//===----------------------------------------------------------------------===//

void AllocConstraints::clear() { allocations.clear(); }

LogicalResult AllocConstraints::insert(Allocation alloc) {
  auto it = allocations.lower_bound(alloc);

  // There's an overlap if the next allocation starts before our end.
  if (it != allocations.end() && it->begin < alloc.end)
    return failure();

  // There's an overlap if the previous allocation ends after our begin.
  if (it != allocations.begin()) {
    if (std::prev(it)->end > alloc.begin)
      return failure();
  }

  // Succeed if we can insert the allocation.
  return success(allocations.insert(alloc).second);
}

FailureOr<Allocation> AllocConstraints::alloc(LDSAllocNode node,
                                              int64_t startPos) {
  int64_t start = startPos;

  auto getStartAligned = [&](int64_t addr) {
    return ((addr + node.alignment - 1) / node.alignment) * node.alignment;
  };

  for (const Allocation &alloc : allocations) {
    // Check if we can fit before this allocation.
    if (start + node.size <= alloc.begin) {
      Allocation result = {start, start + node.size};
      allocations.insert(result);
      return result;
    }
    start = getStartAligned(alloc.end);
  }

  // Check if we can fit at the end.
  if (start + node.size <= kMaxMemory) {
    Allocation result = {start, start + node.size};
    allocations.insert(result);
    return result;
  }

  return failure();
}

void AllocConstraints::print(raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(allocations, os, [&](const Allocation &alloc) {
    os << "[" << alloc.begin << ", " << alloc.end << ")";
  });
  os << "}";
}

//===----------------------------------------------------------------------===//
// LDSAllocator
//===----------------------------------------------------------------------===//

/// Get the offset of an allocation, or -1 if not set.
static int64_t getOffset(AllocLDSOp allocOp) {
  return allocOp.getOffset() ? static_cast<int64_t>(*allocOp.getOffset()) : -1;
}

LogicalResult LDSAllocator::collectConstraints(NodeId nodeId,
                                               ArrayRef<LDSAllocNode> nodes) {
  for (auto [src, tgt] : graph.edges(nodeId)) {
    int64_t offset = getOffset(nodes[tgt].allocOp);
    if (offset < 0)
      continue;
    // Update the total size to reflect preallocated buffers.
    totalSize = std::max(totalSize, offset + nodes[tgt].size);
    if (failed(constraints.insert({offset, offset + nodes[tgt].size}))) {
      AllocLDSOp allocOp = nodes[tgt].allocOp;
      return allocOp.emitError()
             << "conflicting allocation constraints for LDS buffer of size "
             << nodes[tgt].size << " at offset " << offset;
    }
  }
  return success();
}

LogicalResult LDSAllocator::alloc(LDSAllocNode node) {
  FailureOr<Allocation> alloc = constraints.alloc(node, startPos);
  if (failed(alloc)) {
    return node.allocOp.emitError()
           << "failed to allocate LDS buffer of size " << node.size
           << " with alignment " << node.alignment;
  }
  node.allocOp.setOffset(alloc->begin);
  totalSize = std::max(totalSize, alloc->end);
  return success();
}

LogicalResult LDSAllocator::run(aster::GPUFuncInterface kernOp) {
  llvm::DenseSet<NodeId> visited;
  ArrayRef<LDSAllocNode> nodes = graph.getAllocNodes();
  if (nodes.empty())
    return success();

  // Get the current total size of preallocated buffers to set the starting
  // position for allocation. This ensures we don't allocate over existing
  // buffers.
  startPos = kernOp.getSharedMemorySize();
  assert(startPos >= 0 && "Shared memory size must be non-negative");
  for (auto [i, node] : llvm::enumerate(nodes)) {
    // Skip already visited or allocated nodes.
    if (visited.insert(i).second == false || getOffset(node.allocOp) >= 0)
      continue;

    LDBG() << "Allocating node[" << i << "]: " << node.allocOp;

    // Collect the neighbors constraints.
    constraints.clear();
    if (failed(collectConstraints(i, nodes)))
      return failure();

    LDBG_OS([&](raw_ostream &os) {
      os << "  Initial constraints: ";
      constraints.print(os);
    });

    // Allocate the node.
    if (failed(alloc(node)))
      return failure();

    LDBG_OS([&](raw_ostream &os) {
      os << "  Final constraints: ";
      constraints.print(os);
    });
  }

  // Set the shared memory size on the kernel.
  kernOp.setSharedMemorySize(getTotalSize());
  return success();
}
