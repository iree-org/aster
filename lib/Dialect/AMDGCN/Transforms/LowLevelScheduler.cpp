//===- LowLevelScheduler.cpp - Pre-RA instruction scheduler ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-register-allocation instruction scheduler that models AMD GPU hardware
// execution queues (VALU, XDL, SALU, VMEM, LGKM). Reorders instructions
// within basic blocks to hide issue latency using a greedy algorithm.
//
// This pass operates on SSA IR (pre-regalloc) where data dependencies are
// captured by def-use chains. A separate post-RA scheduler (future work)
// would use ReachingDefinitionsAnalysis for side-effect-based dependencies.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/QueueTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LOWLEVELSCHEDULER
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

#define DEBUG_TYPE "amdgcn-low-level-scheduler"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Dependency DAG (shared infrastructure)
//===----------------------------------------------------------------------===//

struct DAGNode {
  Operation *op;
  QueueType queueType;
  int64_t execLatency;
  SmallVector<DAGNode *, 4> successors;
  int64_t numUnscheduledPreds = 0;
};

struct DependencyDAG {
  /// Nodes indexed by operation. Owns the DAGNode memory.
  DenseMap<Operation *, std::unique_ptr<DAGNode>> nodes;

  DAGNode *getOrCreate(Operation *op) {
    auto &node = nodes[op];
    if (!node) {
      node = std::make_unique<DAGNode>();
      node->op = op;
      node->queueType = classifyOp(op);
      node->execLatency = getExecLatency(op, node->queueType);
    }
    return node.get();
  }

  /// Add a dependency edge: `from` must be scheduled before `to`.
  /// Returns true if the edge was new (incremented pred count).
  bool addEdge(Operation *from, Operation *to) {
    DAGNode *fromNode = getOrCreate(from);
    DAGNode *toNode = getOrCreate(to);
    if (llvm::is_contained(fromNode->successors, toNode))
      return false;
    fromNode->successors.push_back(toNode);
    toNode->numUnscheduledPreds++;
    return true;
  }

  /// Collect all root nodes (no unscheduled predecessors).
  SmallVector<DAGNode *> getRoots() const {
    SmallVector<DAGNode *> roots;
    for (const auto &[op, node] : nodes) {
      if (node->numUnscheduledPreds == 0)
        roots.push_back(node.get());
    }
    return roots;
  }
};

//===----------------------------------------------------------------------===//
// DAG builders -- strategy determines how dependencies are discovered
//===----------------------------------------------------------------------===//

/// Add memory ordering and barrier edges (shared by all strategies).
/// Conservative: all VMEM ops are chained, all LGKM ops are chained.
/// Side-effecting unknown ops act as full scheduling barriers.
static void addMemoryAndBarrierEdges(DependencyDAG &dag, Block &block) {
  Operation *lastVMEMOp = nullptr;
  Operation *lastLGKMOp = nullptr;
  Operation *lastBarrier = nullptr;
  SmallVector<Operation *, 32> opsSinceBarrier;

  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (!dag.nodes.count(&op))
      continue;

    QueueType qt = classifyOp(&op);

    // Memory ordering: chain ops within the same memory domain.
    if (qt == QueueType::VMEM) {
      if (lastVMEMOp)
        dag.addEdge(lastVMEMOp, &op);
      lastVMEMOp = &op;
    } else if (qt == QueueType::LGKM) {
      if (lastLGKMOp)
        dag.addEdge(lastLGKMOp, &op);
      lastLGKMOp = &op;
    }

    // Wait ops are NOT full barriers. Instead, add targeted edges:
    // for each token the wait consumes, find the producing load and
    // add edges from the wait to every in-block user of that load's
    // non-token results. This ensures data consumers come after the
    // wait while allowing independent ops to schedule across it.
    if (auto waitOp = dyn_cast<WaitOp>(op)) {
      for (Value tokenArg : waitOp.getDependencies()) {
        auto *loadOp = tokenArg.getDefiningOp();
        if (!loadOp || loadOp->getBlock() != &block)
          continue;
        // The load's non-token results carry the actual data.
        for (OpResult result : loadOp->getResults()) {
          if (isa<TokenDependencyTypeInterface>(result.getType()))
            continue;
          for (Operation *user : result.getUsers()) {
            if (user->getBlock() == &block && user != &op)
              dag.addEdge(&op, user);
          }
        }
      }
      opsSinceBarrier.push_back(&op);
      continue;
    }

    // All other unknown ops with side effects are full barriers.
    bool isBarrier = (qt == QueueType::Unknown && !isMemoryEffectFree(&op));
    if (isBarrier) {
      for (Operation *prev : opsSinceBarrier)
        dag.addEdge(prev, &op);
      opsSinceBarrier.clear();
      lastBarrier = &op;
      lastVMEMOp = &op;
      lastLGKMOp = &op;
    }

    if (lastBarrier && lastBarrier != &op)
      dag.addEdge(lastBarrier, &op);

    opsSinceBarrier.push_back(&op);
  }
}

/// Build a dependency DAG for pre-regalloc SSA IR.
/// Dependencies come from SSA def-use chains (values flow through results).
static DependencyDAG buildSSADAG(Block &block) {
  DependencyDAG dag;

  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    dag.getOrCreate(&op);

    // SSA data dependencies: if an operand is defined by an op in this block,
    // that defining op must be scheduled first.
    for (Value operand : op.getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (def->getBlock() == &block)
          dag.addEdge(def, &op);
      }
    }
  }

  // Add memory ordering and barrier edges on top of SSA edges.
  addMemoryAndBarrierEdges(dag, block);
  return dag;
}

//===----------------------------------------------------------------------===//
// Greedy scheduler (shared infrastructure)
//===----------------------------------------------------------------------===//

struct ScheduleResult {
  SmallVector<Operation *> schedule;
  SmallVector<int64_t> stallCycles;
  SmallVector<StringRef> stallReasons; // empty string when no stall
};

static FailureOr<ScheduleResult> scheduleBlock(DependencyDAG &dag,
                                               Block &block) {
  if (dag.nodes.empty())
    return ScheduleResult{};

  SmallVector<DAGNode *> readyList = dag.getRoots();
  ScheduleResult result;
  QueueType lastQueueType = QueueType::Unknown;
  int64_t burstCount = 0;
  QueueSimulator sim;

  while (!readyList.empty()) {
    DAGNode *best = nullptr;
    int bestScore = std::numeric_limits<int>::min();

    // Scoring priority:
    // 0. Stall avoidance: if an op would stall, penalize but don't
    //    make it infinitely bad -- interleaving a stalling high-latency
    //    op with a few VALU ops is better than exhausting all VALU first
    // 1. Latency hiding (+execLatency): high-latency ops scheduled early
    // 2. Burst continuity (+100): group same-queue ops to reduce switches
    // 3. Critical path (+successors*10): unblock more downstream work
    for (DAGNode *node : readyList) {
      int score = 0;
      int64_t stall = sim.wouldStall(node->queueType);
      if (stall > 0) {
        // Penalize stalls, but cap so high-latency ops still beat
        // a long sequence of low-latency ops. The cap means "schedule
        // the stalling op after ~N low-latency ops, not after ALL of them".
        int64_t cappedStall = std::min(stall, int64_t{32});
        score -= static_cast<int>(cappedStall) * 10;
      }
      int64_t depth = getQueueDepth(node->queueType);
      if (node->queueType == lastQueueType && burstCount < depth)
        score += 100;
      score += static_cast<int>(node->execLatency);
      score += static_cast<int>(node->successors.size()) * 10;

      if (score > bestScore) {
        bestScore = score;
        best = node;
      }
    }

    assert(best && "ready list was non-empty but no best found");

    int64_t stall = sim.issue(best->queueType, best->execLatency);
    result.schedule.push_back(best->op);
    result.stallCycles.push_back(stall);
    if (stall > 0) {
      // e.g. "vmem full" or "xdl full"
      result.stallReasons.push_back(getQueueName(best->queueType));
    } else {
      result.stallReasons.push_back("");
    }

    if (best->queueType == lastQueueType) {
      burstCount++;
    } else {
      lastQueueType = best->queueType;
      burstCount = 1;
    }

    llvm::erase(readyList, best);

    for (DAGNode *succ : best->successors) {
      assert(succ->numUnscheduledPreds > 0);
      succ->numUnscheduledPreds--;
      if (succ->numUnscheduledPreds == 0)
        readyList.push_back(succ);
    }
  }

  // Verify we scheduled everything (no cycles in the DAG).
  if (result.schedule.size() != dag.nodes.size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "LowLevelScheduler: DAG has " << dag.nodes.size()
               << " nodes but scheduled " << result.schedule.size() << "\n");
    return failure();
  }

  // Apply the schedule: move ops to just before the terminator, in order.
  if (!block.mightHaveTerminator())
    return result;
  Operation *terminator = block.getTerminator();
  for (Operation *op : result.schedule)
    op->moveBefore(terminator);

  return result;
}

//===----------------------------------------------------------------------===//
// Pre-RA pass: uses SSA def-use chains for dependencies
//===----------------------------------------------------------------------===//

struct LowLevelSchedulerPass
    : public amdgcn::impl::LowLevelSchedulerBase<LowLevelSchedulerPass> {
  using Base::Base;

  void runOnOperation() override {
    KernelOp kernel = getOperation();
    for (Block &block : kernel.getBodyRegion()) {
      DependencyDAG dag = buildSSADAG(block);
      auto resultOrFailure = scheduleBlock(dag, block);
      if (failed(resultOrFailure))
        return signalPassFailure();

      if (debugStalls) {
        auto &result = *resultOrFailure;
        auto *ctx = kernel.getContext();
        auto stallAttr = StringAttr::get(ctx, "sched.stall_cycles");
        auto reasonAttr = StringAttr::get(ctx, "sched.stall_reason");
        auto i64Ty = IntegerType::get(ctx, 64);
        for (auto [op, stall, reason] : llvm::zip(
                 result.schedule, result.stallCycles, result.stallReasons)) {
          op->setAttr(stallAttr, IntegerAttr::get(i64Ty, stall));
          if (stall > 0)
            op->setAttr(reasonAttr, StringAttr::get(ctx, reason + " full"));
        }
      }
    }
  }
};

} // namespace
