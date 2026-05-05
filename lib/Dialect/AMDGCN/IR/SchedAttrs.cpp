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
  //
  // The classical "fence" pin -- wait -> every subsequent token producer of
  // the same kind -- is asymmetric:
  //
  //   * LGKM (DS read/write, SMEM): KEEP the pin. A wait between a
  //     ds_write and a ds_read encodes the user's intent that the read
  //     not race past the write (LDS-side memory order); reordering would
  //     break the fence regardless of what ConvertWaits later computes
  //     for the count.
  //
  //   * VMEM (flat / global / buffer): DROP the pin for token-based waits.
  //     Such waits drain memory loads whose data is consumed by the wait's
  //     SSA users (e.g. a ds_write that reads the loaded VGPRs). The SSA
  //     edge already keeps the consumer behind the wait. Pinning every
  //     OTHER subsequent load behind the wait blocks load pipelining --
  //     in the prologue, four SSA-independent loads end up serialized
  //     behind their per-load waits because L_{i+1} is forced after W_i.
  //     ConvertWaits / WaitAnalysis recompute vmcnt at the wait's final
  //     IR position, so allowing later loads to issue first just bumps
  //     the recomputed count to keep the original wait semantic intact.
  //
  // For waits with EXPLICIT vmcnt (i.e. authored or rewritten with a
  // hardware count baked in) we still pin -- the count cannot be
  // recomputed and is sensitive to in-flight depth at the wait's
  // position.
  bool hasExplicitVm = wait.getVmCnt() != WaitOp::kNoWaitCount;
  bool waitsLgkm = wait.getLgkmCnt() != WaitOp::kNoWaitCount;
  for (Value dep : wait.getDependencies()) {
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

    bool producesVM = llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
      return isTokenKind(type, MemoryInstructionKind::Flat);
    });
    bool producesLgkm =
        llvm::any_of(TypeRange(op->getResults()), [&](Type type) {
          return isTokenKind(type, MemoryInstructionKind::Shared) ||
                 isTokenKind(type, MemoryInstructionKind::Constant);
        });
    // VMEM pin only on explicit hardware count; LGKM pin always when the
    // wait covers LGKM (preserves user-intent ds_write -> wait -> ds_read
    // fence semantics).
    if (hasExplicitVm && producesVM)
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

    // Pin every memory *write* across the barrier (LDS, SMEM, flat /
    // global / buffer). `s_barrier` is synchronizing the side effects
    // that become observable AFTER it, so store ordering on either
    // side of the barrier is what we need to preserve. Reads (ds_read,
    // s_load_*, global_load, buffer_load) carry no such fence
    // requirement and benefit from being free to schedule across the
    // barrier -- their data dependencies are already tracked by SSA
    // and by token-based wait deps.
    //
    // SALU and SGPR-producing ops are NOT pinned here either:
    //   * SCC / VCC ordering is captured explicitly by
    //     `addImplicitFlagEdges`.
    //   * SGPR data dependencies are tracked by SSA.
    //   * Pinning SALU / SGPR-out transitively pins the address
    //     calculations of post-barrier global_loads (their addresses
    //     are computed by `s_add_u32` / `s_lshl_b32`), forcing the
    //     loads into a tight burst right after the barrier. The VMEM
    //     queue saturates and the next `vmcnt(0)` pays the full
    //     execution latency. Letting the address-calc SALU bypass the
    //     barrier lets the global_load it feeds bypass too, so the
    //     scheduler can spread loads across the iteration body.
    bool producesWriteToken = llvm::any_of(instOp->getResults(), [](Value res) {
      return isa<WriteTokenType>(res.getType());
    });
    if (producesWriteToken) {
      addEdge(op, i);
      continue;
    }
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
    // Memory return latency on MI300X (~256c is the realistic round-trip
    // for hot-cache loads; a true L2 miss is ~700-1000c, but for hiding
    // analysis we model the typical case).
    return 256;
  case QueueType::LGKM: {
    // ds_write / SMEM store: ~8c. ds_read: ~32c. Detect writes by the
    // presence of a WriteToken result (matches the isLgkmWrite[] flag
    // used by computeScore). Anything that isn't a write is treated as
    // a read.
    for (Value res : op->getResults()) {
      if (isa<WriteTokenType>(res.getType()))
        return 8;
    }
    return 32;
  }
  case QueueType::Unknown:
    return 4;
  }
  llvm_unreachable("unhandled queue type");
}

/// Returns the queue depth (number of in-flight slots).
/// In practice ATT traces show effective queue depth ~4 across all kinds:
/// even though some hardware queues advertise more, throughput collapses
/// once 5+ ops of the same kind are in flight (memory queue saturation,
/// XDL pipe contention). 4 is the working depth for the scheduler.
static int64_t getQueueDepth(QueueType /*qt*/) { return 4; }

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
///
/// Two-rate issue model:
///   * Normal issue: `kIssueCost` (4c) per op -- the issue port can launch
///     a new op into the queue every quad if the queue has room.
///   * Queue full: issue blocks until the oldest in-flight slot completes
///     its `execLatency`. Total advance is just the back-pressure stall
///     (no extra port cost on top -- the port time is absorbed by the
///     stall).
///   * XDL back-to-back: a second MFMA right after another MFMA pays the
///     full `execLatency` (16c or 32c) because the XDL pipe is non-
///     pipelined for chained same-kind issues. This models the
///     "consecutive MFMA bunching" cost we see in traces.
struct QueueSimulator {
  DenseMap<QueueType, SmallVector<int64_t>> slotFreeAt;
  int64_t currentCycle = 0;
  QueueType lastIssuedKind = QueueType::Unknown;

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

  /// Issue an op. Returns stall in hw cycles.
  int64_t issue(QueueType qt, int64_t execLatency) {
    if (qt == QueueType::Unknown)
      return 0;

    auto &slots = slotFreeAt[qt];
    llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });

    int64_t depth = getQueueDepth(qt);
    int64_t stallCycles = 0;
    bool queueWasFull = static_cast<int64_t>(slots.size()) >= depth;
    if (queueWasFull) {
      int64_t earliest = *llvm::min_element(slots);
      stallCycles = std::max(int64_t{0}, earliest - currentCycle);
      currentCycle += stallCycles;
      llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });
    }

    slots.push_back(currentCycle + execLatency);
    // Issue cost: full execLatency for back-to-back XDLs (MFMA pipe is
    // non-pipelined when chained same-kind), kIssueCost otherwise.
    // When the queue was full, the port time is absorbed by stallCycles
    // -- don't double-charge.
    int64_t issueCost = 0;
    if (!queueWasFull) {
      if (qt == QueueType::XDL && lastIssuedKind == QueueType::XDL)
        issueCost = execLatency;
      else
        issueCost = kIssueCost;
    }
    currentCycle += issueCost;
    lastIssuedKind = qt;
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
  // Simulator currentCycle at which the most recent VMEM / LGKM was
  // issued, or INT64_MIN if never. Used by the wait-deferral term to
  // compute "how much of the producer exec latency has been hidden",
  // which is more accurate than counting ops in the recent-kinds window
  // when issue rate is variable (XDL back-to-back charges execLatency,
  // not kIssueCost).
  int64_t lastVmemIssueCycle = std::numeric_limits<int64_t>::min();
  int64_t lastLgkmIssueCycle = std::numeric_limits<int64_t>::min();
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
///
/// `isLgkmWrite` is true for LGKM ops that produce a WriteToken (ds_write,
/// SMEM stores). DS writes have ~2x the issue cost of DS reads on CDNA3 and
/// often serialize against same-bank reads through the LDS arbiter, so they
/// get a heavier per-occurrence burst penalty than DS reads.
static int computeScore(QueueType qt, int64_t stall, const SchedState &s,
                        bool consumerInReady, uint8_t waitKind,
                        bool isLgkmWrite, int64_t currentCycle) {
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

  // 3. Per-kind burst penalty. Each kind has its own lookback window and
  //    per-occurrence penalty, sized to the natural interleaving distance:
  //
  //      VMEM:  lookback 16, penalty 125/occurrence
  //             Each global_load takes ~128c exec but stalls the per-CU
  //             memory queue when packed. Spreading them over a 16-op
  //             (~64c issue time) window keeps the queue healthy. With
  //             base 800, 4 in last-16 drops to 300 (< MFMA 400), 5+
  //             below MFMA. Modest bump from 100 to 125 -- larger bumps
  //             over-spread loads and degrade mean iteration time.
  //
  //      LGKM (read):  lookback 4, penalty 200/occurrence
  //             DS read exec ~16-32c. After 1 DS read in last 4, score
  //             drops to 0 (< VALU 100): avoids back-to-back DS reads.
  //
  //      LGKM (write): lookback 4, penalty 250/occurrence
  //             DS writes have higher issue cost than reads on CDNA3
  //             and often serialize against same-bank reads through the
  //             LDS arbiter. Modest 25% bump over reads drives stronger
  //             alternation between ds_write and other ops without
  //             over-deferring writes when LDS pressure demands them.
  //
  //      XDL:   lookback 2, penalty 350/occurrence
  //             MFMA exec 16-32c. Tight 2-op window because we want strict
  //             alternation: 1 MFMA in last 2 -> 400-350 = 50 < VALU (100)
  //             and < LGKM (200), so a non-MFMA always wins next slot when
  //             available. Consumer-bonused MFMAs (400+200=600) still fire
  //             back-to-back when needed (600-350=250 beats LGKM 200).
  //
  //      other (VALU/SALU/Unknown): no burst penalty -- short-latency
  //             ops, no spreading pressure needed.
  ArrayRef<QueueType> recent(s.recentKinds);
  size_t lookback = 0;
  int perOccPenalty = 0;
  switch (qt) {
  case QueueType::VMEM:
    lookback = 16;
    perOccPenalty = 175;
    break;
  case QueueType::LGKM:
    lookback = 4;
    perOccPenalty = isLgkmWrite ? 250 : 200;
    break;
  case QueueType::XDL:
    lookback = 2;
    perOccPenalty = 350;
    break;
  default:
    break;
  }
  if (lookback > 0) {
    ArrayRef<QueueType> window = recent.take_back(lookback);
    int recentCount = static_cast<int>(llvm::count(window, qt));
    score -= perOccPenalty * recentCount;
  }

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

  // 6. Wait deferral: penalize amdgcn.wait ops proportional to the
  //    amount of stall they would create if fired right now.
  //
  //    A wait stalls until its target counter (vmcnt / lgkmcnt) drops
  //    below the threshold. The drain cost is dominated by how much of
  //    the producer queue's issue-to-completion latency hasn't been
  //    hidden yet. We approximate "hide time" by counting ops issued
  //    since the most recent producer of the waited kind in our 32-deep
  //    window: each issued op costs ~`kIssueCost` cycles, so
  //
  //      remainingStall ~= max(0, queueLatency - opsSince * kIssueCost)
  //
  //    A wait fires "for free" only when remainingStall == 0 (its target
  //    has already returned). Issuing one earlier wastes those cycles
  //    on a hardware stall, so the penalty must be large enough to lose
  //    to ANY useful op (LGKM=200, VMEM=800, XDL=400, even after burst
  //    penalties) and to dwarf the queue-saturation stall (-25*64 =
  //    -1600 max from term 1).
  //
  //    Using the same `25 * cycles` weight as term 1 keeps both stall
  //    contributors in the same units. With kMemoryLatency=2700, a
  //    just-issued vmcnt scores -67500, which is overwhelmingly larger
  //    than any tier bonus -- the wait only fires when nothing useful
  //    is ready, or when the projected stall has been amortized.
  enum : uint8_t { WK_VM = 1, WK_LGKM = 2 };
  if (waitKind != 0 && qt == QueueType::Unknown) {
    constexpr int kStallWeight = 25;
    // Wait-deferral memory latency is intentionally a bit larger than the
    // simulator's VMEM exec latency (256c). The simulator's 256c models
    // queue occupancy / port time -- the rate at which a new load can
    // enter the in-flight pool. The actual HBM round-trip a vmcnt(0)
    // wait must hide is closer to 600-700c on MI300X (HBM3 ~500ns at
    // ~2GHz). Using 256c here under-defers waits and they fire while
    // loads are still returning, creating hardware stalls.
    constexpr int kVmemWaitLatency = 300;
    // ds_read exec latency upper bound (LDS arbiter). Simulator uses
    // the same value, so no decoupling needed for LGKM.
    constexpr int kLgkmExecLatency = 32;
    auto remainingStall = [&](int64_t lastIssueCycle, int execLat) -> int {
      if (lastIssueCycle == std::numeric_limits<int64_t>::min())
        return 0; // No producer ever issued -> wait would fire instantly.
      int64_t hideDone = currentCycle - lastIssueCycle;
      return static_cast<int>(std::max<int64_t>(0, execLat - hideDone));
    };
    if (waitKind & WK_VM)
      score -=
          kStallWeight * remainingStall(s.lastVmemIssueCycle, kVmemWaitLatency);
    if (waitKind & WK_LGKM)
      score -=
          kStallWeight * remainingStall(s.lastLgkmIssueCycle, kLgkmExecLatency);
  }

  // 7. (No explicit VMEM saturation cap.) With queue depth = 4 and exec
  //    latency = 256c, the simulator's `wouldStall` already charges a
  //    huge stall (capped at 64c -> -1600 via term 1) for the 5th+ VMEM,
  //    which is sufficient backpressure. An additional explicit cap
  //    here would double-count the saturation penalty.

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
  // True for LGKM ops that produce a WriteToken of Shared/Constant kind
  // (ds_write, SMEM stores). DS writes have heavier issue cost than reads
  // and often serialize against same-bank reads, so they get a heavier
  // per-occurrence burst penalty in computeScore.
  SmallVector<bool> isLgkmWrite(schedGraph.sizeNodes(), false);
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
    if (queueTypes[i] == QueueType::LGKM) {
      for (Value res : op->getResults()) {
        if (isLgkm(res.getType()) && isa<WriteTokenType>(res.getType())) {
          isLgkmWrite[i] = true;
          break;
        }
      }
    }
  }

  SchedState state;
  QueueSimulator sim;

  // Barrier-aware structural scheduling. Precompute, for each barrier, its
  // transitive predecessor closure in the SchedGraph and the count of
  // memory ops in that closure. Drain mode triggers when the imminent
  // barrier (lowest unscheduled-IR-position barrier) has no unscheduled
  // memory ops in its closure -- i.e., all loads/stores on the path have
  // fired and only pure compute (SALU) and waits remain. In drain mode we
  // skip the score-based loop and force-pick from `ready`: the barrier
  // itself if available, else any node in the closure. This delivers the
  // "barrier-first, then mfmas" pattern: MFMAs (not on the barrier's
  // closure because handleBarrier only pins Ds/SMem/Salu/SGPR-out) are
  // held back through the drain and fire after the barrier, hiding their
  // long execution latency behind the post-barrier loads/ds_reads.
  SmallVector<bool> isBarrier(schedGraph.sizeNodes(), false);
  for (int32_t i = 0; i < schedGraph.sizeNodes(); ++i)
    if (isa<SBarrier>(schedGraph.getOp(i)))
      isBarrier[i] = true;

  auto isMemoryNode = [&](int32_t n) -> bool {
    Operation *op = schedGraph.getOp(n);
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      return instOp.hasAnyProps(
          {InstProp::Ds, InstProp::IsVmem, InstProp::Smem});
    return false;
  };

  // Reverse adjacency for backward BFS over the SchedGraph.
  SmallVector<SmallVector<int32_t>> revAdj(schedGraph.sizeNodes());
  for (int32_t u = 0; u < schedGraph.sizeNodes(); ++u)
    for (const auto &edge : schedGraph.edges(u))
      revAdj[edge.second].push_back(u);

  SmallVector<SmallVector<int32_t>> nodeContainingBarriers(
      schedGraph.sizeNodes());
  DenseMap<int32_t, int32_t> barrierUnscheduledMemPreds;
  DenseMap<int32_t, DenseSet<int32_t>> barrierClosure;

  for (int32_t b = 0; b < schedGraph.sizeNodes(); ++b) {
    if (!isBarrier[b])
      continue;
    DenseSet<int32_t> visited;
    SmallVector<int32_t> worklist;
    for (int32_t pred : revAdj[b])
      if (visited.insert(pred).second)
        worklist.push_back(pred);
    while (!worklist.empty()) {
      int32_t cur = worklist.pop_back_val();
      for (int32_t pred : revAdj[cur])
        if (visited.insert(pred).second)
          worklist.push_back(pred);
    }
    int32_t memCount = 0;
    for (int32_t n : visited) {
      nodeContainingBarriers[n].push_back(b);
      if (isMemoryNode(n))
        memCount++;
    }
    barrierUnscheduledMemPreds[b] = memCount;
    barrierClosure[b] = std::move(visited);
  }

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

    // Drain mode: structural barrier-first rule. Find the imminent
    // (lowest unscheduled-IR-position) barrier; if its predecessor closure
    // has no remaining unscheduled memory ops, only pure compute and waits
    // remain on the path -- skip score-based selection and pick a closure
    // member from `ready` (or the barrier itself when ready). This keeps
    // MFMAs (not in any barrier's closure) from being scheduled while
    // a barrier is "almost-ready", letting the barrier fire first and the
    // MFMAs hide behind the post-barrier loads/ds_reads.
    int32_t imminent = -1;
    for (auto &kv : barrierUnscheduledMemPreds) {
      if (imminent < 0 || kv.first < imminent)
        imminent = kv.first;
    }
    if (imminent >= 0 && barrierUnscheduledMemPreds[imminent] == 0) {
      // Phase 1: barrier itself if in ready.
      for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
        if (ready[i] == imminent) {
          bestIdx = i;
          break;
        }
      }
      // Phase 2: any closure member in ready.
      if (bestIdx < 0) {
        const DenseSet<int32_t> &closure = barrierClosure[imminent];
        for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
          if (closure.contains(ready[i])) {
            bestIdx = i;
            break;
          }
        }
      }
      // If nothing path-related is ready right now, fall through to scoring.
    }

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

    if (bestIdx < 0) {
      for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
        int32_t nodeId = ready[i];
        QueueType qt = queueTypes[nodeId];
        int64_t stall = sim.wouldStall(qt);
        bool consumerInReady =
            (qt == QueueType::XDL) && hasXdlConsumerInReady(nodeId);
        int score =
            computeScore(qt, stall, state, consumerInReady, waitKind[nodeId],
                         isLgkmWrite[nodeId], sim.currentCycle);

        if (score > bestScore ||
            (score == bestScore && nodeId < ready[bestIdx])) {
          bestScore = score;
          bestIdx = i;
        }
      }
    }

    assert(bestIdx >= 0 && "schedFn must select a ready node");

    int32_t chosen = ready[bestIdx];
    QueueType chosenQt = queueTypes[chosen];
    int64_t stall = sim.issue(chosenQt, execLatencies[chosen]);
    stallCycles.push_back(stall);
    stallReasons.push_back(stall > 0 ? getQueueName(chosenQt) : "");

    // Update barrier-closure tracking:
    //   - Issuing a barrier removes it from the imminent-barrier table.
    //   - Issuing a memory op decrements the memory-pred count for every
    //     barrier whose closure contains it.
    if (isBarrier[chosen]) {
      barrierUnscheduledMemPreds.erase(chosen);
    } else if (isMemoryNode(chosen)) {
      for (int32_t b : nodeContainingBarriers[chosen]) {
        auto it = barrierUnscheduledMemPreds.find(b);
        if (it != barrierUnscheduledMemPreds.end() && it->second > 0)
          it->second--;
      }
    }

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
      state.lastLgkmIssueCycle = sim.currentCycle;
      // Activate the remaining-consumers counter for this LGKM so that
      // XDL siblings each draw from the deadline pool.
      if (auto it = state.lgkmTotalConsumers.find(chosen);
          it != state.lgkmTotalConsumers.end())
        state.lgkmRemainingConsumers[chosen] = it->second;
      break;
    case QueueType::VMEM:
      state.outstandingVmem++;
      state.lastVmemIssueCycle = sim.currentCycle;
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
