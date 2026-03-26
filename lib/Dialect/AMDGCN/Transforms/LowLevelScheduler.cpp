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

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
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

//===----------------------------------------------------------------------===//
// Queue classification
//===----------------------------------------------------------------------===//

enum class QueueType : uint8_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };

/// Parse sched.queue attr: "valu", "xdl", "salu", "vmem", "lgkm".
// TODO: put this in instruction definition directly in tablegen.
static std::optional<QueueType> parseQueueAttr(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("sched.queue");
  if (!attr)
    return std::nullopt;
  return StringSwitch<std::optional<QueueType>>(attr.getValue())
      .Case("valu", QueueType::VALU)
      .Case("xdl", QueueType::XDL)
      .Case("salu", QueueType::SALU)
      .Case("vmem", QueueType::VMEM)
      .Case("lgkm", QueueType::LGKM)
      .Default(std::nullopt);
}

// TODO: put this in instruction definition directly in tablegen.
static QueueType classifyOp(Operation *op) {
  // sched.queue overrides InstProp classification (useful for test_inst).
  if (auto qt = parseQueueAttr(op))
    return *qt;

  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp)
    return QueueType::Unknown;
  const InstMetadata *md = instOp.getInstMetadata();
  if (!md)
    return QueueType::Unknown;

  // SOPP (s_waitcnt, s_barrier, branches) must be scheduling barriers.
  if (md->hasProp(InstProp::Sopp))
    return QueueType::Unknown;
  if (md->hasProp(InstProp::Dsmem))
    return QueueType::LGKM;
  if (md->hasProp(InstProp::Smem))
    return QueueType::LGKM;
  if (md->hasProp(InstProp::IsVmem))
    return QueueType::VMEM;
  // Check before VALU: MFMA ops carry both Mma and IsValu props.
  if (md->hasAnyProps({InstProp::Mma, InstProp::ScaledMma}))
    return QueueType::XDL;
  if (md->hasProp(InstProp::Salu))
    return QueueType::SALU;
  if (md->hasProp(InstProp::IsValu))
    return QueueType::VALU;

  return QueueType::Unknown;
}

/// Returns exec latency in hw cycles. sched.exec_latency overrides defaults.
/// Note: these are all approximations atm.
// TODO: put this in instruction definition directly in tablegen.
static int64_t getExecLatency(Operation *op, QueueType qt) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("sched.exec_latency"))
    return attr.getInt();
  switch (qt) {
  case QueueType::VALU:
    return 4;
  case QueueType::XDL:
    return 16;
  case QueueType::SALU:
    return 4;
  case QueueType::VMEM:
    return 128;
  case QueueType::LGKM:
    return 32;
  case QueueType::Unknown:
    return 4;
  }
  llvm_unreachable("unhandled queue type");
}

/// Returns the queue depth (number of in-flight slots).
/// Note: these are all approximations atm.
/// VMEM is 2-deep (shared per CU across ~4 waves).
/// All per-SIMD queues are 8-deep.
// TODO: put this in instruction definition directly in tablegen.
static int64_t getQueueDepth(QueueType qt) {
  switch (qt) {
  case QueueType::VMEM:
    return 2;
  default:
    return 8;
  }
}

/// Issue cost in hardware cycles (1 quad = 4 hw cycles).
static constexpr int64_t kIssueCost = 4;

static StringRef getQueueName(QueueType qt) {
  switch (qt) {
  case QueueType::VALU:
    return "valu";
  case QueueType::XDL:
    return "xdl";
  case QueueType::SALU:
    return "salu";
  case QueueType::VMEM:
    return "vmem";
  case QueueType::LGKM:
    return "lgkm";
  case QueueType::Unknown:
    return "unknown";
  }
  llvm_unreachable("unhandled queue type");
}

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

/// Add serialization edges for i1-producing ops within a block.
/// All i1 producers (lsir.cmpi, lsir.cmpf, etc.) write to the same physical
/// flag register (SCC or VCC), creating implicit WAW/WAR hazards invisible
/// to SSA. We ensure ALL consumers of one i1 producer are scheduled before
/// the next i1 producer fires.
static void addI1SerializationEdges(DependencyDAG &dag, Block &block) {
  SmallVector<Operation *> prevI1Consumers;

  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    bool producesI1 = false;
    for (OpResult result : op.getResults()) {
      if (result.getType().isInteger(1)) {
        producesI1 = true;
        break;
      }
    }
    if (!producesI1)
      continue;

    // ALL consumers of the previous i1 producer must be scheduled before
    // this i1 producer to avoid clobbering the flag register.
    for (Operation *consumer : prevI1Consumers)
      dag.addEdge(consumer, &op);

    prevI1Consumers.clear();
    bool hasConsumers = false;
    for (OpResult result : op.getResults()) {
      if (!result.getType().isInteger(1))
        continue;
      for (Operation *user : result.getUsers()) {
        if (user->getBlock() == &block) {
          prevI1Consumers.push_back(user);
          hasConsumers = true;
        }
      }
    }

    // Dead i1 producers: the producer itself is the serialization barrier.
    if (!hasConsumers)
      prevI1Consumers.push_back(&op);
  }
}

/// Build a dependency DAG for pre-regalloc SSA IR.
/// Dependencies come from SSA def-use chains (including token chains),
/// targeted wait->consumer edges, memory-only barriers for count-based
/// waits and s_barrier, and i1 flag register serialization.
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

  // Total-order chain: preserves program order for all non-trivial ops.
  {
    Operation *lastOp = nullptr;
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      if (!dag.nodes.count(&op))
        continue;
      bool isTrivial =
          isa<AllocaOp, MakeRegisterRangeOp, SplitRegisterRangeOp>(op) ||
          op.hasTrait<OpTrait::ConstantLike>();
      if (isTrivial)
        continue;
      if (lastOp)
        dag.addEdge(lastOp, &op);
      lastOp = &op;
    }
  }

  // i1 serialization: all i1 producers (cmpi/cmpf) write to the same
  // physical flag register (SCC/VCC), so their lifetimes must not overlap.
  addI1SerializationEdges(dag, block);

  return dag;
}

//===----------------------------------------------------------------------===//
// Queue simulator -- tracks slot occupancy per queue to detect stalls
//===----------------------------------------------------------------------===//

/// Models the hardware queue state for stall detection.
/// Each queue has `capacity` slots; issuing an op occupies one slot for
/// `execLatency` cycles. A stall occurs when all slots are busy.
struct QueueSimulator {
  DenseMap<QueueType, SmallVector<int64_t, 8>> slotFreeAt;
  int64_t currentCycle = 0;
  QueueSimulator() = default;

  /// Query how many hw cycles issuing to `qt` would stall.
  int64_t wouldStall(QueueType qt) const {
    if (qt == QueueType::Unknown)
      return 0;
    auto it = slotFreeAt.find(qt);
    if (it == slotFreeAt.end())
      return 0;
    int64_t depth = getQueueDepth(qt);
    int64_t occupied = 0;
    for (int64_t t : it->second) {
      if (t > currentCycle)
        occupied++;
    }
    if (occupied < depth)
      return 0;
    int64_t earliest = *llvm::min_element(it->second);
    return std::max(int64_t{0}, earliest - currentCycle);
  }

  /// Issue an op. Returns stall in hw cycles (always a multiple of 4).
  int64_t issue(QueueType qt, int64_t execLatency) {
    if (qt == QueueType::Unknown)
      return 0;

    auto &slots = slotFreeAt[qt];
    llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });

    int64_t depth = getQueueDepth(qt);
    int64_t stallCycles = 0;
    if (static_cast<int64_t>(slots.size()) >= depth) {
      int64_t earliest = *llvm::min_element(slots);
      stallCycles = std::max(int64_t{0}, earliest - currentCycle);
      currentCycle += stallCycles;
      llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });
    }

    slots.push_back(currentCycle + execLatency);
    currentCycle += kIssueCost;
    return stallCycles;
  }
};

//===----------------------------------------------------------------------===//
// Greedy scheduler
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

    // Scoring: stall avoidance + latency-aware interleaving.
    //
    // 1. Stall avoidance: penalize ops that would stall on a full queue.
    // 2. Interleaving bonus: prefer switching queues to overlap execution
    //    (DS/VMEM -> XDL/VALU -> wait pattern).
    for (DAGNode *node : readyList) {
      int score = 0;
      int64_t stall = sim.wouldStall(node->queueType);
      if (stall > 0) {
        int64_t cappedStall = std::min(stall, int64_t{32});
        score -= static_cast<int>(cappedStall) * 10;
      }
      // Interleaving: prefer switching queues to overlap execution.
      if (node->queueType != lastQueueType && burstCount > 0)
        score += 50;

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
// DAG dump (for testing DAG construction independently of scheduling)
//===----------------------------------------------------------------------===//

/// Print a concise identifier for an operation: first result SSA name + op
/// mnemonic. E.g. "%dest_res (load)" or "%vdst0_res (vop2)".
static void printOpId(llvm::raw_ostream &os, Operation *op) {
  if (op->getNumResults() > 0) {
    op->getResult(0).printAsOperand(os, OpPrintingFlags());
  } else {
    // No results -- use the op name (e.g. "wait", "end_kernel").
    os << "<<" << op->getName() << ">>";
  }
  os << " (" << op->getName() << ")";
}

/// Dump the DAG edges and node properties in a FileCheck-friendly format.
/// Output is buffered per kernel to avoid interleaving when MLIR
/// parallelizes across KernelOps.
static void dumpDAG(DependencyDAG &dag, Block &block, KernelOp kernel) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << "DAG for kernel @" << kernel.getSymName() << " {\n";

  // Print nodes in block order for deterministic output.
  for (Operation &op : block) {
    auto it = dag.nodes.find(&op);
    if (it == dag.nodes.end())
      continue;
    DAGNode *node = it->second.get();
    os << "  node: ";
    printOpId(os, &op);
    os << " [queue=" << getQueueName(node->queueType) << "]\n";
    for (DAGNode *succ : node->successors) {
      os << "    -> ";
      printOpId(os, succ->op);
      os << "\n";
    }
  }
  os << "}\n";
  llvm::errs() << os.str();
}

//===----------------------------------------------------------------------===//
// Pre-RA pass: uses SSA def-use chains for dependencies
//===----------------------------------------------------------------------===//

struct LowLevelSchedulerPass
    : public amdgcn::impl::LowLevelSchedulerBase<LowLevelSchedulerPass> {
  using Base::Base;

  void runOnOperation() override {
    KernelOp kernel = getOperation();
    auto *ctx = kernel.getContext();

    // Pre-condition: all function calls must be inlined before scheduling.
    if (!skipPrecondition) {
      auto allInlined = AllInlinedAttr::get(ctx);
      if (failed(normalform::verifyNormalForm(kernel, allInlined,
                                              /*emitDiagnostics=*/true)))
        return signalPassFailure();
    }

    bool failed_ = false;
    kernel->walk([&](Block *block) {
      if (failed_)
        return;
      DependencyDAG dag = buildSSADAG(*block);
      if (dumpDag) {
        dumpDAG(dag, *block, kernel);
        return;
      }
      auto resultOrFailure = scheduleBlock(dag, *block);
      if (failed(resultOrFailure)) {
        failed_ = true;
        return;
      }
      if (debugStalls) {
        auto &result = *resultOrFailure;
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
    });
    if (failed_)
      return signalPassFailure();
  }
};

} // namespace
