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
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
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
// DAG builders -- strategy determines how dependencies are discovered
//===----------------------------------------------------------------------===//

/// Add serialization edges for i1-producing ops within a block.
/// All i1 producers (lsir.cmpi, lsir.cmpf, etc.) write to the same physical
/// flag register (SCC or VCC), creating implicit WAW/WAR hazards invisible
/// to SSA or side effect. We ensure ALL consumers of one i1 producer are
/// scheduled before the next i1 producer fires.
static void addI1SerializationEdges(SchedGraph &graph, Block &block) {
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
      graph.addEdge(consumer, &op);

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

// Uses getInDegree + edges() rather than SchedGraph::topologicalSched because
// the greedy scorer picks one node at a time with QueueSimulator state.
static FailureOr<ScheduleResult> scheduleBlock(const SchedGraph &graph,
                                               ArrayRef<QueueType> queueTypes,
                                               ArrayRef<int64_t> execLatencies,
                                               Block &block) {
  if (graph.sizeNodes() == 0)
    return ScheduleResult{};

  SmallVector<int32_t> inDegree = graph.getInDegree();

  // Collect roots (in-degree 0).
  SmallVector<int32_t> readyList;
  for (int32_t i = 0, e = graph.sizeNodes(); i < e; ++i) {
    if (inDegree[i] == 0)
      readyList.push_back(i);
  }

  ScheduleResult result;
  QueueType lastQueueType = QueueType::Unknown;
  int64_t burstCount = 0;
  QueueSimulator sim;

  while (!readyList.empty()) {
    int32_t best = -1;
    int bestScore = std::numeric_limits<int>::min();

    // Scoring: stall avoidance + latency-aware interleaving.
    //
    // 1. Stall avoidance: penalize ops that would stall on a full queue.
    // 2. Interleaving bonus: prefer switching queues to overlap execution
    //    (DS/VMEM -> XDL/VALU -> wait pattern).
    for (int32_t nodeId : readyList) {
      int score = 0;
      int64_t stall = sim.wouldStall(queueTypes[nodeId]);
      if (stall > 0) {
        int64_t cappedStall = std::min(stall, int64_t{32});
        score -= static_cast<int>(cappedStall) * 10;
      }
      // Interleaving: prefer switching queues to overlap execution.
      if (queueTypes[nodeId] != lastQueueType && burstCount > 0)
        score += 50;

      if (score > bestScore) {
        bestScore = score;
        best = nodeId;
      }
    }

    assert(best >= 0 && "ready list was non-empty but no best found");

    int64_t stall = sim.issue(queueTypes[best], execLatencies[best]);
    result.schedule.push_back(graph.getOp(best));
    result.stallCycles.push_back(stall);
    if (stall > 0) {
      result.stallReasons.push_back(getQueueName(queueTypes[best]));
    } else {
      result.stallReasons.push_back("");
    }

    if (queueTypes[best] == lastQueueType) {
      burstCount++;
    } else {
      lastQueueType = queueTypes[best];
      burstCount = 1;
    }

    llvm::erase(readyList, best);

    for (const auto &edge : graph.edges(best)) {
      int32_t succ = edge.second;
      assert(inDegree[succ] > 0);
      --inDegree[succ];
      if (inDegree[succ] == 0)
        readyList.push_back(succ);
    }
  }

  // Verify we scheduled everything (no cycles in the DAG).
  if (static_cast<int>(result.schedule.size()) != graph.sizeNodes()) {
    LLVM_DEBUG(llvm::dbgs()
               << "LowLevelScheduler: DAG has " << graph.sizeNodes()
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
    auto *ctx = kernel.getContext();

    // Pre-condition: all function calls must be inlined before scheduling.
    auto allInlined = AllInlinedAttr::get(ctx);
    if (failed(normalform::verifyNormalForm(kernel, allInlined,
                                            /*emitDiagnostics=*/true)))
      return signalPassFailure();

    // Set up GraphBuilder (SSA deps, wait tokens, barriers).
    auto graphAttr = ValueSchedulerAttr::get(ctx);
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
    SchedAnalysis analysis(kernel, solver, domInfo, getAnalysisManager());
    if (failed(graphAttr.initializeAnalyses(analysis)))
      return signalPassFailure();
    if (analysis.shouldRunDataflowAnalyses()) {
      if (failed(solver.initializeAndRun(kernel)))
        return signalPassFailure();
    }

    bool failed_ = false;
    kernel->walk([&](Block *block) {
      if (failed_)
        return;

      // Build dependency graph via GraphBuilder.
      FailureOr<SchedGraph> graphOrFailure =
          graphAttr.createGraph(block, analysis);
      if (failed(graphOrFailure)) {
        failed_ = true;
        return;
      }
      SchedGraph &graph = *graphOrFailure;

      // i1 serialization: GraphBuilder doesn't handle flag register hazards.
      addI1SerializationEdges(graph, *block);
      graph.compress();

      // Build side arrays for queue types and exec latencies.
      SmallVector<QueueType> queueTypes(graph.sizeNodes());
      SmallVector<int64_t> execLatencies(graph.sizeNodes());
      for (auto [i, op] : llvm::enumerate(graph.getOps())) {
        queueTypes[i] = classifyOp(op);
        execLatencies[i] = getExecLatency(op, queueTypes[i]);
      }

      auto resultOrFailure =
          scheduleBlock(graph, queueTypes, execLatencies, *block);
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
