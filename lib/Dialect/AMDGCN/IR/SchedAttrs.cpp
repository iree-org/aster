//===- SchedAttrs.cpp - AMDGCN scheduling attribute implementations -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-sched"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// ValueSchedulerAttr - SchedGraphAttrInterface
//===----------------------------------------------------------------------===//

namespace {
struct GraphBuilder {
  GraphBuilder(Block *block, const DataFlowSolver &solver)
      : block(block), solver(const_cast<DataFlowSolver &>(solver)) {
    assert(block && "expected a valid block");
  }

  /// Run the graph builder on the given block, adding edges between operations.
  LogicalResult run(SchedGraph &graph);

private:
  /// Build the SSA dependencies for the graph.
  void buildSSADeps(SchedGraph &graph);

  /// Build the non-SSA dependencies for the graph.
  void buildNonSSADeps(SchedGraph &graph);

  /// Handle a wait operation.
  void handleWaitOp(SchedGraph &graph, int64_t pos, WaitOp wait);

  /// Handle a barrier operation. With barriers we must add dependencies before
  /// and after if the operation affects SALU or SGPRs.
  void handleBarrier(SchedGraph &graph, int64_t pos, Operation *barrier);

  /// Add serialization edges for i1-producing ops within a block.
  void addI1SerializationEdges(SchedGraph &graph);

  Block *block;
  SmallVector<int64_t> syncPoints;
  DataFlowSolver &solver;
};
} // namespace

LogicalResult
ValueSchedulerAttr::initializeAnalyses(SchedAnalysis &analysis) const {
  // Load the wait analysis.
  analysis.getSolver().load<WaitAnalysis>(analysis.getDomInfo());
  analysis.setRunDataflowAnalyses();
  return success();
}

LogicalResult GraphBuilder::run(SchedGraph &graph) {
  buildSSADeps(graph);
  buildNonSSADeps(graph);
  addI1SerializationEdges(graph);
  return success();
}

void GraphBuilder::buildSSADeps(SchedGraph &graph) {
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    LDBG() << "Processing operation: " << i << " "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());

    bool hasEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
                      op->hasTrait<MemoryEffectOpInterface::Trait>();

    // If the operation has no side-effect we need to treat it as a possible
    // sync point. Same for non-pure operations.
    if ((!hasEffects || !mlir::isPure(op)) &&
        !isa<LoadOp, StoreOp, AllocaOpInterface>(op)) {
      LDBG() << "Adding sync point: " << i;
      syncPoints.push_back(i);
    }

    ValueRange deps = op->getOperands();

    // Add edges for the dependencies.
    for (Value operand : deps) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getBlock() == block)
        graph.addEdge(producer, op);
    }
  }
}

void GraphBuilder::buildNonSSADeps(SchedGraph &graph) {
  ArrayRef<Operation *> ops = graph.getOps();
  for (int64_t &i : syncPoints) {
    Operation *op = ops[i];
    if (auto waitOp = dyn_cast<WaitOp>(op)) {
      handleWaitOp(graph, i, waitOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
    if (auto barrierOp = dyn_cast<SBarrier>(op)) {
      handleBarrier(graph, i, barrierOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
  }

  // Erase all the processed sync points.
  llvm::erase(syncPoints, -1);

  // If there are no sync points, return.
  if (syncPoints.empty())
    return;

  // Add edges between all the ops before a sync point and the sync point, and
  // between the sync point and all the ops after it.
  for (auto [i, syncPoint] : llvm::enumerate(syncPoints)) {
    int64_t prevSyncPoint = i > 0 ? syncPoints[i - 1] : 0;
    int64_t nextSyncPoint =
        i < (syncPoints.size() - 1) ? syncPoints[i + 1] + 1 : ops.size();
    for (int64_t point = prevSyncPoint; point < syncPoint; point++)
      graph.addEdge(point, syncPoint);
    for (int64_t point = syncPoint + 1; point < nextSyncPoint; point++)
      graph.addEdge(syncPoint, point);
  }
}

void GraphBuilder::handleWaitOp(SchedGraph &graph, int64_t pos, WaitOp wait) {
  // Get the wait state.
  const WaitState *state =
      solver.lookupState<WaitState>(solver.getProgramPointAfter(wait));
  assert(state && "expected valid wait state");

  // Collect all the operations that sync at this point.
  SetVector<Operation *> waitedOps;
  for (const TokenState &token : state->waitOpInfo->waitedTokens) {
    if (!token.getToken())
      continue;
    Operation *op = token.getToken().getDefiningOp();
    if (!op || op->getBlock() != block)
      continue;
    waitedOps.insert(op);
  }
  for (const TokenState &token : state->waitOpInfo->impliedTokens) {
    if (!token.getToken())
      continue;
    Operation *op = token.getToken().getDefiningOp();
    if (!op || op->getBlock() != block)
      continue;
    waitedOps.insert(op);
  }

  // Add edges for the waited operations.
  for (Operation *op : waitedOps)
    graph.addEdge(op, wait);

  auto isTokenKind = [](Type type, MemoryInstructionKind kind) {
    if (auto tokType = dyn_cast<ReadTokenType>(type))
      return tokType.getKind() == kind;
    if (auto tokType = dyn_cast<WriteTokenType>(type))
      return tokType.getKind() == kind;
    return false;
  };

  // Add edges for the operations that are after this wait operation that wait
  // on the same tokens.
  bool waitsVM = wait.getVmCnt() != WaitOp::kNoWaitCount;
  bool waitsLgkm = wait.getLgkmCnt() != WaitOp::kNoWaitCount;
  for (Value dep : wait.getDependencies()) {
    if (isTokenKind(dep.getType(), MemoryInstructionKind::Flat))
      waitsVM = true;
    if (isTokenKind(dep.getType(), MemoryInstructionKind::Shared) ||
        isTokenKind(dep.getType(), MemoryInstructionKind::Constant))
      waitsLgkm = true;
  }
  for (Operation *op : graph.getOps().drop_front(pos + 1)) {
    ValueRange operands = op->getOperands();
    for (Value operand : operands) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || producer->getBlock() != block)
        continue;
      if (waitedOps.contains(producer))
        graph.addEdge(wait, op);
    }

    // Prevent operations producing tokens moving before a wait operation.
    bool producesVM = llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
      return isTokenKind(type, MemoryInstructionKind::Flat);
    });
    bool producesLgkm =
        llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
          return isTokenKind(type, MemoryInstructionKind::Shared) ||
                 isTokenKind(type, MemoryInstructionKind::Constant);
        });
    if (waitsVM && producesVM)
      graph.addEdge(wait, op);
    if (waitsLgkm && producesLgkm)
      graph.addEdge(wait, op);
  }
}

void GraphBuilder::handleBarrier(SchedGraph &graph, int64_t pos,
                                 Operation *barrier) {
  // Helper function to add an edge between an operation and the barrier.
  auto addEdge = [&](Operation *op, int64_t i) {
    if (i < pos)
      graph.addEdge(op, barrier);
    if (i > pos)
      graph.addEdge(barrier, op);
  };

  // Iterate over all the operations in the graph.
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    // Skip itself.
    if (op == barrier)
      continue;

    auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);

    // If there's no interface, add an edge from the barrier to the operation.
    if (!instOp || instOp.getOpCode() == OpCode::Invalid) {
      if (!isPure(op))
        addEdge(op, i);
      continue;
    }

    // If the operation has any SALU or SMEM properties, add an edge.
    if (instOp.hasAnyProps({InstProp::Salu, InstProp::Smem, InstProp::Dsmem})) {
      addEdge(op, i);
      continue;
    }

    // If the operation has any SGPR outputs, add an edge.
    bool hasSGPROut = llvm::any_of(instOp->getResults(), [](Value result) {
      return isa<SGPRType>(result.getType());
    });
    if (hasSGPROut)
      addEdge(op, i);
  }
}

void GraphBuilder::addI1SerializationEdges(SchedGraph &graph) {
  SmallVector<Operation *> prevI1Consumers;

  for (Operation &op : *block) {
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

    for (Operation *consumer : prevI1Consumers)
      graph.addEdge(consumer, &op);

    prevI1Consumers.clear();
    bool hasConsumers = false;
    for (OpResult result : op.getResults()) {
      if (!result.getType().isInteger(1))
        continue;
      for (Operation *user : result.getUsers()) {
        if (user->getBlock() == block) {
          prevI1Consumers.push_back(user);
          hasConsumers = true;
        }
      }
    }

    if (!hasConsumers)
      prevI1Consumers.push_back(&op);
  }
}

FailureOr<SchedGraph>
ValueSchedulerAttr::createGraph(Block *block,
                                const SchedAnalysis &analysis) const {
  SchedGraph graph(block);
  GraphBuilder builder(block, analysis.getSolver());
  if (failed(builder.run(graph)))
    return failure();
  graph.compress();
  return graph;
}

//===----------------------------------------------------------------------===//
// InstPropLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t InstPropLabelerAttr::getLabel(Operation *op, int32_t,
                                      const SchedGraph &) const {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return -1;
  ArrayRef<InstProp> matcher = getInstMatcher();
  if (matcher.empty())
    return getStage();
  if (!instOp.hasAnyProps(matcher))
    return -1;
  return getStage();
}

//===----------------------------------------------------------------------===//
// OpCodeLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t OpCodeLabelerAttr::getLabel(Operation *op, int32_t,
                                    const SchedGraph &) const {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return -1;
  ArrayRef<OpCode> matcher = getOpCodeMatcher();
  if (matcher.empty())
    return getStage();
  OpCode opcode = instOp.getOpCode();
  if (!llvm::is_contained(matcher, opcode))
    return -1;
  return getStage();
}

//===----------------------------------------------------------------------===//
// LatencyPipelinerSchedAttr - SchedBuilderAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult
LatencyPipelinerSchedAttr::createSched(const SchedGraph &schedGraph,
                                       SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  int32_t limit = getSchedLimit();
  if (limit <= 0)
    limit = std::numeric_limits<int32_t>::max();

  auto schedFn = [&](ArrayRef<int32_t> ready, SetVector<int32_t> &indices) {
    bool hasZeroLabel = false;
    // Phase 1: schedule label-0 ops immediately.
    for (auto [i, node] : llvm::enumerate(ready)) {
      if (schedGraph.getLabel(node) == 0) {
        hasZeroLabel = true;
        indices.insert(i);
      }
    }

    // Early exit if we scheduled any label-0 ops.
    if (hasZeroLabel)
      return;

    // Phase 2: interleaved highest-latency scheduling.
    using IntPair = std::pair<int32_t, int32_t>;
    SmallVector<IntPair> sortedReady;
    sortedReady =
        llvm::map_to_vector(llvm::enumerate(ready), [&](auto indexNode) {
          auto [i, node] = indexNode;
          return std::make_pair(node, static_cast<int32_t>(i));
        });
    llvm::sort(sortedReady, [&](IntPair a, IntPair b) {
      int32_t labelA = schedGraph.getLabel(a.first);
      int32_t labelB = schedGraph.getLabel(b.first);
      // Sort descending by label; break ties by ascending node ID.
      if (labelA != labelB)
        return labelA > labelB;
      return a.first < b.first;
    });
    if (sortedReady.empty())
      return;

    int32_t numToAdd = std::min<int32_t>(limit, ready.size());
    int32_t currLabel = schedGraph.getLabel(sortedReady.front().first);
    int32_t idx = 1;
    // Schedule the highest-latency op.
    indices.insert(sortedReady.front().second);
    --numToAdd;

    // Circular scheduler, always trying to schedule ops with different labels.
    while (numToAdd > 0) {
      // Skip already-inserted ops and ops with the same label as the last
      // scheduled one, advancing the circular index.
      if (indices.contains(sortedReady[idx].second) ||
          currLabel == schedGraph.getLabel(sortedReady[idx].first)) {
        idx = (idx + 1) % sortedReady.size();

        // If we looped, reset the label so in the next iteration we will
        // schedule the highest-latency op among the remaining ones.
        if (idx == 0)
          currLabel = std::numeric_limits<int32_t>::min();
        continue;
      }
      // Schedule the op and update the current label.
      indices.insert(sortedReady[idx].second);
      --numToAdd;
      currLabel = schedGraph.getLabel(sortedReady[idx].first);
      idx = (idx + 1) % sortedReady.size();
      if (idx == 0)
        currLabel = std::numeric_limits<int32_t>::min();
    }
  };

  return schedGraph.topologicalSched(schedFn, sched);
}

//===----------------------------------------------------------------------===//
// LowLevelSchedulerAttr - SchedBuilderAttrInterface
//===----------------------------------------------------------------------===//

namespace {
enum class QueueType : uint8_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };
} // namespace

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
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return QueueType::Unknown;

  // SOPP (s_waitcnt, s_barrier, branches) must be scheduling barriers.
  if (instOp.hasProp(InstProp::Sopp))
    return QueueType::Unknown;
  if (instOp.hasProp(InstProp::Dsmem))
    return QueueType::LGKM;
  if (instOp.hasProp(InstProp::Smem))
    return QueueType::LGKM;
  if (instOp.hasProp(InstProp::IsVmem))
    return QueueType::VMEM;
  // Check before VALU: MFMA ops carry both Mma and IsValu props.
  if (instOp.hasAnyProps({InstProp::Mma, InstProp::ScaledMma}))
    return QueueType::XDL;
  if (instOp.hasProp(InstProp::Salu))
    return QueueType::SALU;
  if (instOp.hasProp(InstProp::IsValu))
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

namespace {
//===----------------------------------------------------------------------===//
// Queue simulator -- tracks slot occupancy per queue to detect stalls
//===----------------------------------------------------------------------===//

/// Models the hardware queue state for stall detection.
/// Each queue has `capacity` slots; issuing an op occupies one slot for
/// `execLatency` cycles. A stall occurs when all slots are busy.
struct QueueSimulator {
  DenseMap<QueueType, SmallVector<int64_t>> slotFreeAt;
  int64_t currentCycle = 0;

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
} // namespace

LogicalResult
LowLevelSchedulerAttr::createSched(const SchedGraph &schedGraph,
                                   SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  if (schedGraph.getOps().empty())
    return success();

  // Classify operations into hardware queues and compute their execution
  // latencies.
  SmallVector<QueueType> queueTypes(schedGraph.sizeNodes());
  SmallVector<int64_t> execLatencies(schedGraph.sizeNodes());
  for (auto [i, op] : llvm::enumerate(schedGraph.getOps())) {
    queueTypes[i] = classifyOp(op);
    execLatencies[i] = getExecLatency(op, queueTypes[i]);
  }

  // Track the last scheduled queue type and the number of consecutive
  // operations on the same queue.
  QueueType lastQueueType = QueueType::Unknown;
  int64_t burstCount = 0;
  QueueSimulator sim;

  // Track stall cycles and reasons for each operation.
  SmallVector<int64_t> stallCycles;
  SmallVector<StringRef> stallReasons;

  // Greedy pick driven by SchedGraph::topologicalSched. When scores tie, prefer
  // the smallest node id for a stable order. Assigns indices to the scheduled
  // operations.
  auto schedFn = [&](ArrayRef<int32_t> ready, SetVector<int32_t> &indices) {
    int32_t bestIdx = -1;
    int bestScore = std::numeric_limits<int>::min();
    bool hasImmSchedOp = false;

    // Scoring: stall avoidance + latency-aware interleaving.
    //
    // 1. Stall avoidance: penalize ops that would stall on a full queue.
    // 2. Interleaving bonus: prefer switching queues to overlap execution
    //    (DS/VMEM -> XDL/VALU -> wait pattern).
    for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
      Operation *op = schedGraph.getOp(ready[i]);
      // Always schedule all trivial operations first.
      if (isa<AllocaOpInterface, MakeRegisterRangeOp, SplitRegisterRangeOp>(
              op) ||
          op->hasTrait<OpTrait::ConstantLike>()) {
        indices.insert(i);
        stallCycles.push_back(0);
        stallReasons.push_back("");
        hasImmSchedOp = true;
        continue;
      }
      if (hasImmSchedOp)
        continue;

      // Compute the score for the current operation.
      int32_t nodeId = ready[i];
      int score = 0;
      int64_t stall = sim.wouldStall(queueTypes[nodeId]);
      if (stall > 0) {
        int64_t cappedStall = std::min(stall, int64_t{32});
        score -= static_cast<int>(cappedStall) * 10;
      }
      // Interleaving: prefer switching queues to overlap execution.
      if (queueTypes[nodeId] != lastQueueType && burstCount > 0)
        score += 50;

      if (score > bestScore ||
          (score == bestScore && nodeId < ready[bestIdx])) {
        bestScore = score;
        bestIdx = i;
      }
    }

    // If we have already scheduled all trivial operations, return. This allows
    // to see farther in the graph.
    if (hasImmSchedOp)
      return;

    assert(bestIdx >= 0 && "schedFn must select a ready node");

    // Record stall cycles and reasons.
    int32_t chosen = ready[bestIdx];
    int64_t stall = sim.issue(queueTypes[chosen], execLatencies[chosen]);
    stallCycles.push_back(stall);
    stallReasons.push_back(stall > 0 ? getQueueName(queueTypes[chosen]) : "");

    // Update the last scheduled queue type and burst count.
    if (queueTypes[chosen] == lastQueueType) {
      burstCount++;
    } else {
      lastQueueType = queueTypes[chosen];
      burstCount = 1;
    }

    // Add the best operation to the scheduled operations.
    indices.insert(bestIdx);
  };

  // Topologically schedule the operations.
  if (failed(schedGraph.topologicalSched(schedFn, sched)))
    return failure();

  // If not annotating stalls, return success.
  if (!getAnnotateStalls())
    return success();

  assert(stallCycles.size() == sched.size() &&
         "sched and stallCycles must have the same size");
  assert(stallReasons.size() == sched.size() &&
         "sched and stallReasons must have the same size");

  MLIRContext *ctx = schedGraph.getOp(0)->getContext();
  auto stallAttr = StringAttr::get(ctx, "sched.stall_cycles");
  auto reasonAttr = StringAttr::get(ctx, "sched.stall_reason");
  auto i64Ty = IntegerType::get(ctx, 64);

  // Annotate each scheduled op with stall cycles and reasons in schedule order.
  for (size_t i = 0, n = sched.size(); i < n; ++i) {
    Operation *op = schedGraph.getOp(sched[i]);
    int64_t stall = stallCycles[i];
    StringRef reason = stallReasons[i];
    op->setAttr(stallAttr, IntegerAttr::get(i64Ty, stall));
    if (stall > 0)
      op->setAttr(reasonAttr, StringAttr::get(ctx, (reason + " full").str()));
  }
  return success();
}
