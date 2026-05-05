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

  /// Add explicit edges to capture implicit SCC and VCC dependencies that
  /// are not surfaced via SSA. Per-flag, walk the block in IR order and
  /// add an edge from the LAST producer in a producer-block to the FIRST
  /// consumer in the next consumer-block (and the symmetric edge for the
  /// next consumer-block -> producer-block transition). Within a block of
  /// only-writers or only-readers, no edges are added: those ops are free
  /// to reorder amongst themselves.
  void addImplicitFlagEdges(SchedGraph &graph);

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
  addImplicitFlagEdges(graph);
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
    //
    // SOP2 SALU arithmetic ops (s_lshl_b32, s_add_u32, s_and_b32, ...) are
    // excluded from sync points: their explicit dst result is tracked via
    // SSA, and the implicit SCC / VCC writes/reads are captured by
    // addImplicitFlagEdges below.
    bool isReorderableSop2 = false;
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      isReorderableSop2 = instOp.hasAnyProps({InstProp::Sop2});
    if ((!hasEffects || !mlir::isPure(op)) && !isReorderableSop2 &&
        !isa<LoadOpInterface, StoreOpInterface, AllocaOpInterface>(op)) {
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
    if (instOp.hasAnyProps({InstProp::Salu, InstProp::Smem, InstProp::Ds})) {
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

/// Classify how `op` interacts with a flag (SCC or VCC) by walking its
/// AMDGCNInstOpInterface outs/ins operand ranges. `Reads` is set if any
/// `ins` operand has the matching flag type; `Writes` is set if any `outs`
/// operand does. Ops without AMDGCNInstOpInterface are treated as neither
/// (their flag interaction is not modeled here).
template <typename FlagType>
static void classifyFlagInteraction(Operation *op, bool &reads, bool &writes) {
  reads = false;
  writes = false;
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp)
    return;
  for (Value v : instOp.getInstOuts()) {
    if (isa<FlagType>(v.getType())) {
      writes = true;
      break;
    }
  }
  for (Value v : instOp.getInstIns()) {
    if (isa<FlagType>(v.getType())) {
      reads = true;
      break;
    }
  }
}

/// Per-flag (SCC, VCC) pass that adds explicit edges to capture implicit
/// flag dependencies that SSA does not surface (the flag is written via
/// a DPS alloca operand rather than as an SSA result).
///
/// Walk the block in IR order. Maintain two running sets:
///   - writeRun: writers since the last reader (or block start)
///   - readRun:  readers since the last writer (or block start)
/// For each W -> R transition, add edges from EVERY writer in writeRun to
/// the new reader (so every writer is pinned before every reader in the
/// next read-run, but writers can still reorder amongst themselves).
/// Symmetric for R -> W. Ops within the same run get no edges between
/// them: the scheduler is free to reorder write-only ops amongst
/// themselves and read-only ops amongst themselves. An op that BOTH
/// reads and writes (e.g. s_addc_u32 reads carry-in and writes
/// carry-out) acts as both a reader and writer at its position.
template <typename FlagType>
static void addFlagEdgesForType(SchedGraph &graph, Block *block) {
  SmallVector<Operation *> writeRun;
  SmallVector<Operation *> readRun;
  for (Operation &op : *block) {
    bool reads = false;
    bool writes = false;
    classifyFlagInteraction<FlagType>(&op, reads, writes);
    if (!reads && !writes)
      continue;
    bool wasReadPhase = !readRun.empty();
    if (reads) {
      for (Operation *w : writeRun)
        graph.addEdge(w, &op);
    }
    if (writes) {
      for (Operation *r : readRun)
        graph.addEdge(r, &op);
      if (wasReadPhase) {
        // R -> W transition: end the read-run and start a fresh
        // write-run rooted at this op (older writers are no longer
        // "active" -- the readers consumed their values).
        readRun.clear();
        writeRun.clear();
      }
    }
    if (reads)
      readRun.push_back(&op);
    if (writes)
      writeRun.push_back(&op);
  }
}

void GraphBuilder::addImplicitFlagEdges(SchedGraph &graph) {
  addFlagEdgesForType<SCCType>(graph, block);
  addFlagEdgesForType<VCCType>(graph, block);
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
// LGKM covers all LDS reads/writes and SMEM loads. Reads and writes have
// different raw latencies (~32c vs ~8c) but in practice the scheduler treats
// them identically for priority and burst bookkeeping -- the 4-bit lgkmcnt
// counter aggregates both, so distinguishing the two is not actionable.
enum class QueueType : uint8_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };
} // namespace

/// Parse sched.queue attr: "valu", "xdl", "salu", "vmem", "lgkm".
/// Legacy "lgkm_r" / "lgkm_w" remain accepted for backward-compat with
/// existing test_inst attrs but both map to the same `LGKM` bucket.
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
      .Case("lgkm_r", QueueType::LGKM)
      .Case("lgkm_w", QueueType::LGKM)
      .Default(std::nullopt);
}

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
  // Any LDS op or SMEM load -> LGKM bucket.
  if (instOp.hasAnyProps({InstProp::Ds, InstProp::Smem}))
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

/// Per-opcode XDL exec latency from CDNA3 ISA Table 28 (MI300 manual p.42).
/// Returns 0 if the opcode is not an XDL instruction we model specifically.
static int64_t getXdlExecLatency(OpCode op) {
  switch (op) {
  // 4-pass: 16-cycle MFMAs (16x16x16 and 16x16x32 family).
  case OpCode::V_MFMA_F32_16X16X16_F16:
  case OpCode::V_MFMA_F32_16X16X16_BF16:
  case OpCode::V_MFMA_F16_16X16X16_F16:
  case OpCode::V_MFMA_F32_16X16X32_F16:
  case OpCode::V_MFMA_F32_16X16X32_BF16:
  case OpCode::V_MFMA_F32_16X16X32_FP8_FP8:
  case OpCode::V_MFMA_F32_16X16X32_FP8_BF8:
  case OpCode::V_MFMA_F32_16X16X32_BF8_FP8:
  case OpCode::V_MFMA_F32_16X16X32_BF8_BF8:
    return 16;
  // 8-pass: 32-cycle MFMAs (32x32x8 / 32x32x16 family).
  case OpCode::V_MFMA_F32_32X32X8_F16:
  case OpCode::V_MFMA_F32_32X32X16_F16:
  case OpCode::V_MFMA_F32_32X32X16_BF16:
    return 32;
  default:
    return 0;
  }
}

/// Returns exec latency in hw cycles. sched.exec_latency overrides defaults.
/// Citations refer to AMD CDNA3 (MI300) Instruction Set Architecture manual.
static int64_t getExecLatency(Operation *op, QueueType qt) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("sched.exec_latency"))
    return attr.getInt();
  switch (qt) {
  case QueueType::VALU:
    return 4;
  case QueueType::XDL: {
    // Per-opcode latency (Table 28, p.42). Default 16 covers the 4-pass case.
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
      if (int64_t lat = getXdlExecLatency(instOp.getOpCode()))
        return lat;
    }
    return 16;
  }
  case QueueType::SALU:
    return 4;
  case QueueType::VMEM:
    return 128;
  case QueueType::LGKM:
    // Mid-range covering both LDS reads (~32c) and writes (~8c). The
    // scheduler treats them as one bucket; this is the issue-cycle
    // estimate used by the QueueSimulator.
    return 16;
  case QueueType::Unknown:
    return 4;
  }
  llvm_unreachable("unhandled queue type");
}

/// Returns the queue depth (number of in-flight slots).
/// VMEM has ~16 outstanding loads in the per-wave load buffer (matches ATT
/// observation; ISA Section 4.4 does not publish a hardware cap, so this
/// number is microarchitectural). All per-SIMD queues are 8-deep.
static int64_t getQueueDepth(QueueType qt) {
  switch (qt) {
  case QueueType::VMEM:
    return 16;
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

//===----------------------------------------------------------------------===//
// Scheduler state -- carried across schedFn calls for deficit-counter scoring
//===----------------------------------------------------------------------===//

/// State carried across schedFn calls. Counts that drive the deficit-term
/// scoring. `outstandingLgkm` is a *deadline* counter, not in-flight latency:
/// it counts unconsumed LGKM loads, decremented when a consumer XDL is
/// issued. This is what "MFMA must drain LDS reads" measures.
struct SchedState {
  int64_t outstandingLgkm = 0;
  int64_t outstandingVmem = 0;
  // Sliding window of the last (up to) `kRecentKindsWindow` issued queue
  // kinds, oldest at front. Drives the last-4 burst penalty (per-occurrence
  // ramp), the VALU<->DS adjacency penalty (8-cycle hardware NOP), and the
  // VMEM density limit.
  static constexpr int64_t kVmemInWindowLimit = 4;
  static constexpr int64_t kRecentKindsWindow = 32;
  SmallVector<QueueType, kRecentKindsWindow> recentKinds;
  // For each XDL node, the list of LGKM predecessors (static, populated
  // once at pre-cache time from the SchedGraph). May contain duplicates if
  // the same LGKM is referenced by multiple operands. In practice only
  // LDS reads / SMEM ever feed XDLs; LDS writes never appear here.
  DenseMap<int32_t, SmallVector<int32_t, 4>> xdlLgkmPreds;
  // For each LGKM node, the total count of XDL successors (static).
  DenseMap<int32_t, int32_t> lgkmTotalConsumers;
  // For each LGKM that has been issued and still has unconsumed XDL
  // successors, the remaining count. Empty before the LGKM fires; removed
  // once all consumers fire. The size of this map is `outstandingLgkm`.
  DenseMap<int32_t, int32_t> lgkmRemainingConsumers;
};
} // namespace

/// Pure scoring function: priority-ordered alternation with a per-occurrence
/// burst penalty.
///
/// Priority order (highest first): mfma > global_load > ds_read > ds_write
/// > other. The bonus differential between consecutive tiers is 100.
///
/// Burst penalty: count occurrences of `qt` in the last 4 issued kinds and
/// subtract `kPerOccurrencePenalty * count`. Linear ramp drives alternation:
/// after a couple of same-kind ops in the window the burst penalty exceeds
/// any tier gap and the next pick prefers a different priority tier.
/// Penalty is symmetric across same-kind candidates, so when no other kind
/// is ready a same-kind op still wins -- the "unless there are no other
/// ones ready" exception.
///
/// `waitKind` is non-zero only for amdgcn.wait ops; layered on top of the
/// "other" tier it bumps low-stall wait_vm above LGKM and below VMEM,
/// matching the previous wait-hide policy.
static int computeScore(QueueType qt, int64_t stall, const SchedState &s,
                        bool consumerInReady, uint8_t waitKind) {
  static constexpr int kPerOccurrencePenalty = 250;
  int score = 0;

  // 1. Stall avoidance: every queue saturation cycle costs 25 points,
  //    capped at 64 cycles so a single deeply-stalled op cannot starve
  //    another priority tier permanently.
  if (stall > 0) {
    int64_t capped = std::min(stall, int64_t{64});
    score -= 25 * static_cast<int>(capped);
  }

  // 2. Priority bonus by kind.
  switch (qt) {
  case QueueType::VMEM:
    score += 800;
    break;
  case QueueType::XDL:
    score += 400;
    break;
  case QueueType::LGKM:
    score += 200;
    break;
  default:
    score += 100;
    break;
  }

  // 3. Linear burst penalty: each occurrence of `qt` in the last 4 issued
  //    kinds contributes -kPerOccurrencePenalty. Ramps 0 / -250 / -500 /
  //    -750 / -1000 for 0..4 occurrences -- drives alternation while still
  //    letting same-kind win when no other kind is ready (penalty is
  //    symmetric across same-kind candidates).
  ArrayRef<QueueType> recent(s.recentKinds);
  ArrayRef<QueueType> last4 = recent.take_back(4);
  int recentCount = static_cast<int>(llvm::count(last4, qt));
  score -= kPerOccurrencePenalty * recentCount;

  // 3a. S/VALU <-> LGKM adjacency penalty: VALU/SALU adjacent to a DS op
  //     (in either direction) requires an 8-cycle hardware NOP. Bias the
  //     scheduler away from this pairing whenever something else is ready.
  QueueType prevKind = recent.empty() ? QueueType::Unknown : recent.back();
  if ((qt == QueueType::VALU && prevKind == QueueType::LGKM) ||
      (qt == QueueType::LGKM && prevKind == QueueType::VALU))
    score -= 400;
  if ((qt == QueueType::SALU && prevKind == QueueType::LGKM) ||
      (qt == QueueType::LGKM && prevKind == QueueType::SALU))
    score -= 400;

  // 3b. VMEM density cap: more than 6 global_load operations in a window
  //     of 32 instructions saturates memory bandwidth -- subsequent
  //     vmcnt(0) waits stall on memory throughput, not just queue
  //     occupancy. Heavy penalty (-1500) discourages issuing the 7th VMEM
  //     until older ones age out of the window.
  if (qt == QueueType::VMEM) {
    int vmemInWindow = static_cast<int>(llvm::count(recent, QueueType::VMEM));
    if (vmemInWindow >= SchedState::kVmemInWindowLimit)
      score -= 1500;
  }

  // 4. Kickstart loads: at schedule start (no ops issued yet), prime the
  //    load pipeline by lifting LGKM / VMEM above XDL so the first MFMA
  //    hides behind a load already in flight.
  if (recent.empty() && (qt == QueueType::LGKM || qt == QueueType::VMEM))
    score += 300;

  // 5. Deadline-driven MFMA hiding: when an XDL's LGKM producer is
  //    issued and waiting on this XDL to drain it, fire it now.
  if (qt == QueueType::XDL && consumerInReady)
    score += 200;

  // 6. Wait-aware bonus for amdgcn.wait ops (which classify as Unknown).
  //    Low-stall wait_vm jumps to between LGKM (200) and VMEM (800);
  //    low-stall wait_lgkm sits at LGKM's tier so the in-flight LGKM
  //    pool drains promptly.
  //
  //    "High stall" is gauged by *time since the last issuing op of the
  //    waited queue*, not by in-flight depth. A vmcnt drain fired right
  //    after its own producing load has in-flight depth = 1 yet still
  //    pays the full ~2700c memory latency: the memory hasn't returned
  //    yet, so vmcnt won't decrement. Counting ops in the recent window
  //    since the last VMEM (resp. LGKM) is a better proxy for "the
  //    drain will be cheap" -- we want enough MFMAs / addr math between
  //    the last load and the wait to amortize the latency.
  //
  //    Thresholds are heuristic: VMEM exec latency is ~128c, MFMA is
  //    16-32c; 8 ops since the last VMEM (~128c of issue+exec time)
  //    means the oldest load is plausibly returning. LGKM exec latency
  //    is ~16c, so 4 ops is enough.
  constexpr size_t kVmemHideDistance = 8;
  constexpr size_t kLgkmHideDistance = 4;
  enum : uint8_t { WK_VM = 1, WK_LGKM = 2 };
  if (waitKind != 0 && qt == QueueType::Unknown) {
    auto opsSince = [&](QueueType q) -> size_t {
      // Distance from the back of `recent` to the most recent `q`.
      // Returns recent.size() if `q` is not in the window (= "far").
      for (size_t i = 0; i < recent.size(); ++i) {
        if (recent[recent.size() - 1 - i] == q)
          return i;
      }
      return recent.size();
    };
    bool waitsVmStallHigh =
        (waitKind & WK_VM) && opsSince(QueueType::VMEM) < kVmemHideDistance;
    bool waitsLgkmStallHigh =
        (waitKind & WK_LGKM) && opsSince(QueueType::LGKM) < kLgkmHideDistance;
    if ((waitKind == WK_VM) && !waitsVmStallHigh)
      score += 200; // 100 (other) + 200 = 300, above LGKM (200)
    else if ((waitKind & WK_LGKM) && !waitsLgkmStallHigh)
      score += 100; // 100 (other) + 100 = 200, at LGKM tier
  }

  // 7. VMEM saturation cap: VMCNT is 4-bit and each VMEM op has ~128c
  //    latency. Stacking too many in flight makes a later vmcnt(0)
  //    stall thousands of cycles. Cap at 4 outstanding to match the
  //    "at most 4 in flight" rule.
  if (qt == QueueType::VMEM && s.outstandingVmem >= 4)
    score -= 250;

  return score;
}

LogicalResult
LowLevelSchedulerAttr::createSched(const SchedGraph &schedGraph,
                                   SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  if (schedGraph.getOps().empty())
    return success();

  // Classify operations into hardware queues and compute their execution
  // latencies. Also pre-classify amdgcn.wait ops by which counter(s) they
  // gate so the priority-order fill rule can prefer wait-vmcnt over
  // wait-lgkmcnt and skip them when their estimated stall is too high.
  SmallVector<QueueType> queueTypes(schedGraph.sizeNodes());
  SmallVector<int64_t> execLatencies(schedGraph.sizeNodes());
  // Bit 0 = waits on vmcnt (flat tokens); bit 1 = waits on lgkmcnt
  // (shared/constant tokens). 0 = not a wait op.
  enum : uint8_t { WK_NotWait = 0, WK_VM = 1, WK_LGKM = 2, WK_Both = 3 };
  SmallVector<uint8_t> waitKind(schedGraph.sizeNodes(), WK_NotWait);
  auto isFlat = [](Type type) {
    if (auto rt = dyn_cast<ReadTokenType>(type))
      return rt.getKind() == MemoryInstructionKind::Flat;
    if (auto wt = dyn_cast<WriteTokenType>(type))
      return wt.getKind() == MemoryInstructionKind::Flat;
    return false;
  };
  auto isLgkm = [](Type type) {
    auto k = MemoryInstructionKind::Shared;
    auto kc = MemoryInstructionKind::Constant;
    if (auto rt = dyn_cast<ReadTokenType>(type))
      return rt.getKind() == k || rt.getKind() == kc;
    if (auto wt = dyn_cast<WriteTokenType>(type))
      return wt.getKind() == k || wt.getKind() == kc;
    return false;
  };
  for (auto [i, op] : llvm::enumerate(schedGraph.getOps())) {
    queueTypes[i] = classifyOp(op);
    execLatencies[i] = getExecLatency(op, queueTypes[i]);
    if (auto waitOp = dyn_cast<WaitOp>(op)) {
      bool wvm = waitOp.getVmCnt() != WaitOp::kNoWaitCount;
      bool wlgkm = waitOp.getLgkmCnt() != WaitOp::kNoWaitCount;
      for (Value dep : waitOp.getDependencies()) {
        if (isFlat(dep.getType()))
          wvm = true;
        if (isLgkm(dep.getType()))
          wlgkm = true;
      }
      waitKind[i] = (wvm ? WK_VM : 0) | (wlgkm ? WK_LGKM : 0);
    }
  }

  SchedState state;
  QueueSimulator sim;

  // Pre-cache the LGKM-R <-> XDL adjacency. Record ALL XDL successors per
  // LGKM-R (not just the first) so the deadline-counter scoring keeps
  // working when one ds_read fans out to multiple MFMA consumers.
  for (int32_t nodeId = 0; nodeId < schedGraph.sizeNodes(); ++nodeId) {
    if (queueTypes[nodeId] != QueueType::LGKM)
      continue;
    int32_t totalConsumers = 0;
    for (const auto &edge : schedGraph.edges(nodeId)) {
      int32_t succId = edge.second;
      assert(succId >= 0 && succId < schedGraph.sizeNodes() &&
             "edge successor out of range");
      if (queueTypes[succId] == QueueType::XDL) {
        state.xdlLgkmPreds[succId].push_back(nodeId);
        totalConsumers++;
      }
    }
    if (totalConsumers > 0)
      state.lgkmTotalConsumers[nodeId] = totalConsumers;
  }

  // Track stall cycles and reasons for each scheduled op.
  SmallVector<int64_t> stallCycles;
  SmallVector<StringRef> stallReasons;

  // Greedy pick driven by SchedGraph::topologicalSched. When scores tie, prefer
  // the smallest node id for a stable order. Assigns indices to the scheduled
  // operations.
  auto schedFn = [&](ArrayRef<int32_t> ready, SetVector<int32_t> &indices) {
    int32_t bestIdx = -1;
    int bestScore = std::numeric_limits<int>::min();
    bool hasImmSchedOp = false;

    // Trivial-op fast path: schedule all alloca/make_register/constant ops
    // immediately so the scheduler sees the real frontier.
    for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
      Operation *op = schedGraph.getOp(ready[i]);
      if (isa<AllocaOpInterface, MakeRegisterRangeOp, SplitRegisterRangeOp>(
              op) ||
          op->hasTrait<OpTrait::ConstantLike>()) {
        indices.insert(i);
        stallCycles.push_back(0);
        stallReasons.push_back("");
        hasImmSchedOp = true;
      }
    }
    if (hasImmSchedOp)
      return;

    // Detect whether an XDL op consumes any LGKM that's been issued and
    // still has unconsumed successors. This is the deadline signal that
    // drives the +200 bonus -- it stays true for ALL siblings sharing the
    // same LGKM producer, not just the first.
    auto hasXdlConsumerInReady = [&](int32_t xdlNode) {
      auto it = state.xdlLgkmPreds.find(xdlNode);
      if (it == state.xdlLgkmPreds.end())
        return false;
      for (int32_t lgkm : it->second) {
        if (state.lgkmRemainingConsumers.count(lgkm))
          return true;
      }
      return false;
    };

    for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
      int32_t nodeId = ready[i];
      QueueType qt = queueTypes[nodeId];
      int64_t stall = sim.wouldStall(qt);
      bool consumerInReady =
          (qt == QueueType::XDL) && hasXdlConsumerInReady(nodeId);
      int score =
          computeScore(qt, stall, state, consumerInReady, waitKind[nodeId]);

      if (score > bestScore ||
          (score == bestScore && nodeId < ready[bestIdx])) {
        bestScore = score;
        bestIdx = i;
      }
    }

    assert(bestIdx >= 0 && "schedFn must select a ready node");

    int32_t chosen = ready[bestIdx];
    QueueType chosenQt = queueTypes[chosen];
    int64_t stall = sim.issue(chosenQt, execLatencies[chosen]);
    stallCycles.push_back(stall);
    stallReasons.push_back(stall > 0 ? getQueueName(chosenQt) : "");

    // Update closure state. Push the chosen kind into the sliding-4 window;
    // drop the oldest if the window is already full.
    state.recentKinds.push_back(chosenQt);
    if (state.recentKinds.size() >
        static_cast<size_t>(SchedState::kRecentKindsWindow))
      state.recentKinds.erase(state.recentKinds.begin());
    switch (chosenQt) {
    case QueueType::XDL:
      // For each LGKM predecessor of this XDL: decrement its remaining-
      // consumers counter. When the last consumer fires, drop the LGKM
      // from `lgkmRemainingConsumers` and decrement `outstandingLgkm`.
      // This keeps the deadline pressure on while siblings of the just-
      // issued XDL are still in flight.
      if (auto it = state.xdlLgkmPreds.find(chosen);
          it != state.xdlLgkmPreds.end()) {
        for (int32_t lgkm : it->second) {
          auto rit = state.lgkmRemainingConsumers.find(lgkm);
          if (rit == state.lgkmRemainingConsumers.end())
            continue; // LGKM not yet issued
          rit->second--;
          if (rit->second <= 0) {
            state.lgkmRemainingConsumers.erase(rit);
            state.outstandingLgkm =
                std::max(int64_t{0}, state.outstandingLgkm - 1);
          }
        }
      }
      break;
    case QueueType::LGKM:
      state.outstandingLgkm++;
      // Activate the remaining-consumers counter for this LGKM so that
      // XDL siblings each draw from the deadline pool.
      if (auto it = state.lgkmTotalConsumers.find(chosen);
          it != state.lgkmTotalConsumers.end())
        state.lgkmRemainingConsumers[chosen] = it->second;
      break;
    case QueueType::VMEM:
      state.outstandingVmem++;
      break;
    default:
      break;
    }

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

  // Annotate each scheduled op with stall cycles and reasons.
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
