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
#include "mlir/Analysis/SliceAnalysis.h"
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
    //
    // Exclude SOP2 SALU arithmetic ops (s_add_*, ...) arithmetic ops from sync
    // points. The implicit read/write effects on architectural registers SCC
    // are captured by the RAW/WAR edges added in buildNonSSADeps.
    // Treating as fences would serialize independent arithmetic chains.
    bool isReorderableArith = false;
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      isReorderableArith =
          instOp.hasAnyProps({InstProp::Sop2, InstProp::IsValu});
    if ((!hasEffects || !mlir::isPure(op)) && !isReorderableArith &&
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

template <typename RegType>
struct IsModeledArchReg : std::false_type {};
template <>
struct IsModeledArchReg<SCCType> : std::true_type {};
template <>
struct IsModeledArchReg<VCCType> : std::true_type {};

/// Returns the read/write effects of `op` on the architectural register
/// `RegType` (currently SCC or VCC). The register is referenced as a DPS
/// alloca operand: an outs operand of `RegType` means the op writes the
/// register; an ins operand of `RegType` means the op reads it. Ops without
/// AMDGCNInstOpInterface have no architectural-register effects modeled
/// here.
template <typename RegType>
static void getArchRegEffects(Operation *op, bool &reads, bool &writes) {
  static_assert(IsModeledArchReg<RegType>::value,
                "getArchRegEffects only supports modeled architectural "
                "registers (SCC, VCC)");
  reads = false;
  writes = false;
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp)
    return;
  for (Value v : instOp.getInstOuts()) {
    if (isa<RegType>(v.getType())) {
      writes = true;
      break;
    }
  }
  for (Value v : instOp.getInstIns()) {
    if (isa<RegType>(v.getType())) {
      reads = true;
      break;
    }
  }
}

/// Add scheduling edges for the read/write effects on a single architectural
/// register (`RegType`). The register is shared by every op that touches it
/// but flows through DPS alloca operands rather than SSA results, so no SSA
/// def-use chain captures the dependence.
template <typename RegType>
static void addArchRegEffectEdges(SchedGraph &graph, Block *block) {
  static_assert(IsModeledArchReg<RegType>::value,
                "addArchRegEffectEdges only supports modeled architectural "
                "registers (SCC, VCC)");
  SmallVector<Operation *> writers;
  SmallVector<Operation *> readers;
  for (Operation &op : *block) {
    bool reads = false;
    bool writes = false;
    getArchRegEffects<RegType>(&op, reads, writes);
    if (!reads && !writes)
      continue;
    if (reads) {
      // RAW: pin every prior writer before this reader.
      for (Operation *w : writers)
        graph.addEdge(w, &op);
    }
    if (writes) {
      // WAR: pin every prior reader before this writer.
      for (Operation *r : readers)
        graph.addEdge(r, &op);
    }
    if (reads)
      readers.push_back(&op);
    if (writes)
      writers.push_back(&op);
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

  // Add fence edges for the remaining sync points: every preceding op in
  // the segment must precede the sync point, and every following op in the
  // segment must come after.
  if (!syncPoints.empty()) {
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

  // RAW / WAR edges for read/write effects on the architectural registers
  // SCC and VCC.
  addArchRegEffectEdges<SCCType>(graph, block);
  addArchRegEffectEdges<VCCType>(graph, block);
}

static bool isTokenOfKind(Type type, MemoryInstructionKind kind) {
  if (auto rt = dyn_cast<ReadTokenType>(type))
    return rt.getKind() == kind;
  if (auto wt = dyn_cast<WriteTokenType>(type))
    return wt.getKind() == kind;
  return false;
}

static bool isFlatToken(Type type) {
  return isTokenOfKind(type, MemoryInstructionKind::Flat);
}

static bool isLgkmToken(Type type) {
  return isTokenOfKind(type, MemoryInstructionKind::Shared) ||
         isTokenOfKind(type, MemoryInstructionKind::Constant);
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

  // Add edges for the operations that are after this wait operation that wait
  // on the same tokens.
  // Waits with explicit counters have side-effects and must be pinned.
  // Pin all DS ops, do not pin global / flat / constant ops (may be unsafe).
  // TODO: This is a crude approximation, needs proper futures + wait_all + then
  // modeling.
  bool hasExplicitVmCnt = wait.getVmCnt() != WaitOp::kNoWaitCount;
  bool hasExplicitLgkmCnt = wait.getLgkmCnt() != WaitOp::kNoWaitCount;
  for (Value dep : wait.getDependencies()) {
    if (isLgkmToken(dep.getType()))
      hasExplicitLgkmCnt = true;
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

    bool producesVM = llvm::any_of(op->getResultTypes(), isFlatToken);
    bool producesLgkm = llvm::any_of(op->getResultTypes(), isLgkmToken);
    // TODO: This is a crude approximation, needs proper futures + wait_all +
    // then modeling.
    if (hasExplicitVmCnt && producesVM)
      graph.addEdge(wait, op);
    if (hasExplicitLgkmCnt && producesLgkm)
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

    // Pin SALU and SMEM to barriers.
    if (instOp.hasAnyProps({InstProp::Salu, InstProp::Smem})) {
      addEdge(op, i);
      continue;
    }
    // Pin DS and VMEM ops to barriers (s_barrier is a memory fence).
    if (instOp.hasAnyProps({InstProp::Ds, InstProp::IsVmem})) {
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
// LGKM covers all LDS reads/writes and SMEM loads. Reads and writes have
// different raw latencies but in practice the lgkmcnt counter aggregates both,
// so distinguishing the two is not actionable (at least o nCDNA3).
enum class QueueType : int32_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };
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

namespace {
//===----------------------------------------------------------------------===//
// Architecture latencies estimates from execution + scheduling-policy constants
//===----------------------------------------------------------------------===//

struct CDNA3Latencies {
  // -- Exec latencies estimates from execution (hw cycles) --------------------
  int64_t valuExec = 4;
  int64_t saluExec = 4;
  int64_t vmemExec = 128; // fallback for VMEM ops we do not classify
  // -- VMEM load latencies (return path), per-dword-count ---------------------
  // Larger payloads return later because the GMEM->L1 transfer pulls more
  // bytes; the wave is only ready when the last beat lands in VGPRs.
  int64_t vmemLoadDwordExec = 80;
  int64_t vmemLoadDwordx2Exec = 96;
  int64_t vmemLoadDwordx3Exec = 112;
  int64_t vmemLoadDwordx4Exec = 128;
  // -- VMEM store latencies (commit / vmcnt-decrement), per-dword-count -------
  // Stores are assumed to be fire-and-forget into the write FIFO, use lower
  // latency than for loads.
  int64_t vmemStoreDwordExec = 8;
  int64_t vmemStoreDwordx2Exec = 12;
  int64_t vmemStoreDwordx3Exec = 16;
  int64_t vmemStoreDwordx4Exec = 24;
  int64_t xdlExec4Pass = 16;    // 16x16x* MFMA family
  int64_t xdlExec8Pass = 32;    // 32x32x* MFMA family
  int64_t dsReadExec = 32;      // LDS bank arb + return path to VGPR
  int64_t dsWriteExec = 12;     // port-held write completion
  int64_t lgkmDefaultExec = 16; // SMEM / other LGKM
  int64_t unknownExec = 4;

  // -- Issue costs (hw cycles the issue port is held) -------------------------
  int64_t defaultIssueCost = 4;  // wave-scheduler cadence
  int64_t dsWriteIssueCost = 12; // wave-wide ds_write_b128 (1024B / 128B/c)

  // -- Queue depths estimateion (in-flight slots per queue) -------------------
  int64_t vmemQueueDepth = 16;   // per-CU
  int64_t defaultQueueDepth = 8; // per-SIMD

  // -- Score: approx. priority bonuses by kind --------------------------------
  int vmemPriority = 800;
  int xdlPriority = 400;
  int lgkmPriority = 200;
  int otherPriority = 100;

  // -- Score: approx. stall weight + cap --------------------------------------
  int stallWeight = 25;  // points per cycle of queue saturation
  int64_t stallCap = 64; // single-stall ceiling

  // -- Score: approx per-kind burst penalty (lookback, penalty/occ) -----------
  size_t vmemBurstLookback = 16;
  int vmemBurstPenalty = 100;
  size_t lgkmBurstLookback = 4;
  int lgkmBurstPenalty = 200;
  size_t xdlBurstLookback = 2;
  int xdlBurstPenalty = 350;

  // -- Score: VALU/SALU <-> LGKM/VMEM adjacency NOP penalty ------------
  int adjacencyPenalty = 400;

  // -- Score: VMEM density cap inside the recent-kinds window ----------
  int64_t vmemInWindowLimit = 3; // > N VMEM in window -> heavy penalty
  int vmemDensityPenalty = 1500;

  // -- Score: XDL with LGKM-pred consumer in ready set -----------------
  int consumerReadyBonus = 200;

  // -- Score: amdgcn.wait deferral -------------------------------------
  int waitMemoryLatency = 2700; // MI300X DRAM round-trip (conservative)
  int waitLgkmExecLatency = 32; // DS read exec latency upper bound
  int waitHideTimePerOp = 4;    // wave-scheduler cadence approximation

  // -- Score: VMEM in-flight saturation cap (VMCNT is 4 bits) ----------
  // Threshold below the 4-bit cap (15) with headroom for intermixed loads.
  int64_t vmemSaturationThreshold = 8;
  int vmemSaturationPenalty = 250;

  // -- Score: VMEM store bonus (fire-and-forget, no in-kernel consumer) -
  int vmemStoreBonus = 1200;

  // -- Score: VMEM store-feeder bonus (address VALU feeding a store) ----
  // Small enough not to preempt non-chain MFMAs (xdlPriority 400).
  int vmemStoreFeederBonus = 150;

  // -- Score: XDL chain-continuation stall estimate ---------------------
  // RAW on the accumulator (~12-16 cycles, CDNA3 16x16); kept below
  // xdlBurstPenalty so burst stays the dominant XDL-queue pressure.
  int xdlChainStallEstimate = 100;
  size_t xdlChainLookback = 4;
};

/// Preset 1: wider LGKM burst lookback (6) keeps the LGKM penalty active longer
struct CDNA3PresetMfmaHiding : CDNA3Latencies {
  CDNA3PresetMfmaHiding() { lgkmBurstLookback = 6; }
};

} // namespace

/// Active per-architecture latency / policy table.
///
/// `preset` selects a pre-tuned magic-number set:
///   1 = mfma-hiding default
///   2..N = future presets (add a new struct above + a new case here).
static CDNA3Latencies latencies(int preset) {
  switch (preset) {
  case 1:
    return CDNA3PresetMfmaHiding();
  default:
    return CDNA3PresetMfmaHiding();
  }
}

/// Per-opcode XDL exec latency from CDNA3 ISA Table 28 (MI300 manual p.42).
/// Returns 0 if the opcode is not an XDL instruction we model specifically.
static int64_t getXdlExecLatency(OpCode op, const CDNA3Latencies &L) {
  switch (op) {
  // 4-pass: 16-cycle MFMAs (16x16x16 and 16x16x32 family).
  case OpCode::v_mfma_f32_16x16x16_f16:
  case OpCode::v_mfma_f32_16x16x16_bf16:
  case OpCode::v_mfma_f32_16x16x32_f16:
  case OpCode::v_mfma_f32_16x16x32_bf16:
  case OpCode::v_mfma_f32_16x16x32_fp8_fp8:
  case OpCode::v_mfma_f32_16x16x32_fp8_bf8:
  case OpCode::v_mfma_f32_16x16x32_bf8_fp8:
  case OpCode::v_mfma_f32_16x16x32_bf8_bf8:
    return L.xdlExec4Pass;
  // 8-pass: 32-cycle MFMAs (32x32x8 / 32x32x16 family).
  case OpCode::v_mfma_f32_32x32x8_f16:
  case OpCode::v_mfma_f32_32x32x16_f16:
  case OpCode::v_mfma_f32_32x32x16_bf16:
    return L.xdlExec8Pass;
  default:
    return 0;
  }
}

/// Returns exec latency in hw cycles. sched.exec_latency overrides defaults.
static int64_t getExecLatency(Operation *op, QueueType qt,
                              const CDNA3Latencies &L) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("sched.exec_latency"))
    return attr.getInt();
  switch (qt) {
  case QueueType::VALU:
    return L.valuExec;
  case QueueType::XDL:
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      if (int64_t lat = getXdlExecLatency(instOp.getOpCode(), L))
        return lat;
    return L.xdlExec4Pass; // default to 4-pass
  case QueueType::SALU:
    return L.saluExec;
  case QueueType::VMEM: {
    // Pick per-width latency based on op mnemonic suffix.
    StringRef name = op->getName().getStringRef();
    bool isStore =
        isa<GlobalStoreDwordInstOpInterface, BufferStoreInstOpInterface>(op);
    if (name.contains("dwordx4"))
      return isStore ? L.vmemStoreDwordx4Exec : L.vmemLoadDwordx4Exec;
    if (name.contains("dwordx3"))
      return isStore ? L.vmemStoreDwordx3Exec : L.vmemLoadDwordx3Exec;
    if (name.contains("dwordx2"))
      return isStore ? L.vmemStoreDwordx2Exec : L.vmemLoadDwordx2Exec;
    if (name.contains("dword"))
      return isStore ? L.vmemStoreDwordExec : L.vmemLoadDwordExec;
    return L.vmemExec;
  }
  case QueueType::LGKM:
    if (isa<DSWriteInstOpInterface>(op))
      return L.dsWriteExec;
    if (isa<DSReadInstOpInterface>(op))
      return L.dsReadExec;
    return L.lgkmDefaultExec;
  case QueueType::Unknown:
    return L.unknownExec;
  }
  llvm_unreachable("unhandled queue type");
}

/// Queue depth (number of in-flight slots).
static int64_t getQueueDepth(QueueType qt, const CDNA3Latencies &L) {
  // TODO: 4 is an approximation here, it depends on predication + stagger (e.g.
  // in ping-pong or wave-specialized schedules).
  int64_t numSimdActive = 4;
  return qt == QueueType::VMEM ? L.vmemQueueDepth / numSimdActive
                               : L.defaultQueueDepth;
}

/// Per-op issue cost in hw cycles -- how long the issue port is held.
static int64_t getIssueCost(Operation *op, QueueType qt,
                            const CDNA3Latencies &L) {
  if (qt == QueueType::LGKM && isa<DSWriteInstOpInterface>(op))
    return L.dsWriteIssueCost;
  return L.defaultIssueCost;
}

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
// Queue simulator -- tracks slot occupancy per queue to detect stalls
//===----------------------------------------------------------------------===//

namespace {
/// Per-queue slot model: `capacity` slots, each held for `execLatency`
/// cycles; a stall occurs when all slots are busy.
struct QueueSimulator {
  DenseMap<QueueType, SmallVector<int64_t>> slotFreeAt;
  int64_t currentCycle = 0;

  /// Cycles a new issue to `qt` would stall.
  int64_t wouldStall(QueueType qt, const CDNA3Latencies &L) const {
    if (qt == QueueType::Unknown)
      return 0;
    auto it = slotFreeAt.find(qt);
    if (it == slotFreeAt.end())
      return 0;
    int64_t depth = getQueueDepth(qt, L);
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

  /// Issue op; returns stall cycles. `issueCost` = issue-port hold (4c
  /// most ops, 12c ds_write_*).
  int64_t issue(QueueType qt, int64_t execLatency, int64_t issueCost,
                const CDNA3Latencies &L) {
    if (qt == QueueType::Unknown)
      return 0;

    auto &slots = slotFreeAt[qt];
    llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });

    int64_t depth = getQueueDepth(qt, L);
    int64_t stallCycles = 0;
    if (static_cast<int64_t>(slots.size()) >= depth) {
      int64_t earliest = *llvm::min_element(slots);
      stallCycles = std::max(int64_t{0}, earliest - currentCycle);
      currentCycle += stallCycles;
      llvm::erase_if(slots, [&](int64_t t) { return t <= currentCycle; });
    }

    slots.push_back(currentCycle + execLatency);
    currentCycle += issueCost;
    return stallCycles;
  }
};

//===----------------------------------------------------------------------===//
// Scheduler state -- carried across schedFn calls for deficit-counter scoring
//===----------------------------------------------------------------------===//

/// Scheduler state counts drives the deficit-term scoring.
struct SchedState {
  // Bitset for amdgcn.wait gating:
  //   - bit 0 = waits on vmcnt (flat tokens),
  //   - bit 1 = waits on lgkmcnt (shared/constant tokens)
  // Non-wait nodes carry WK_NotWait.
  static constexpr int32_t WK_NotWait = 0;
  static constexpr int32_t WK_VM = 1;
  static constexpr int32_t WK_LGKM = 2;

  int64_t outstandingVmem = 0;

  // Sliding window of the last (up to) `kRecentKindsWindow` issued queue kinds.
  // Drives the burst penalty, the VALU<->DS adjacency penalty, and the VMEM
  // density limit.
  static constexpr int64_t kRecentKindsWindow = 32;
  SmallVector<QueueType, kRecentKindsWindow> recentKinds;
  // Parallel window of node ids -- same lifetime as recentKinds. Used by
  // the XDL chain-continuation check to test whether a candidate MFMA's
  // operand-defining MFMA was issued in the recent window.
  SmallVector<int32_t, kRecentKindsWindow> recentIds;

  // For each XDL node, the prepopulated list of LGKM predecessors.
  DenseMap<int32_t, SmallVector<int32_t, 4>> xdlLgkmPreds;

  // For each LGKM node, the total count of XDL successors.
  DenseMap<int32_t, int32_t> lgkmTotalConsumers;

  // For each LGKM that has been issued and still has unconsumed XDL successors,
  // the remaining count. Empty before the LGKM fires; removed
  // once all consumers fire
  int64_t outstandingLgkm = 0;
  DenseMap<int32_t, int32_t> lgkmRemainingConsumers;

  // Per-op static classification, filled once by `classifyOps()`. Indexed by
  // node id; lifetime spans the entire scheduling run.
  SmallVector<QueueType> queueTypes;
  SmallVector<int64_t> execLatencies;
  SmallVector<int64_t> issueCosts;
  SmallVector<int32_t> waitKind;

  /// Classify every node in `schedGraph`.
  void classifyOps(const SchedGraph &schedGraph, const CDNA3Latencies &L);

  /// Drives the consumer-ready bonus to incentivize ds_read.
  bool hasXdlConsumerInReady(int32_t xdlNode) const;

  /// Update the deadline counters for a freshly issued op.
  void onIssued(int32_t chosen);
};
} // namespace

bool SchedState::hasXdlConsumerInReady(int32_t xdlNode) const {
  auto it = xdlLgkmPreds.find(xdlNode);
  if (it == xdlLgkmPreds.end())
    return false;
  for (int32_t lgkm : it->second) {
    if (lgkmRemainingConsumers.count(lgkm))
      return true;
  }
  return false;
}

void SchedState::onIssued(int32_t chosen) {
  switch (queueTypes[chosen]) {
  case QueueType::XDL: {
    auto preds = xdlLgkmPreds.find(chosen);
    if (preds == xdlLgkmPreds.end())
      break;
    for (int32_t lgkm : preds->second) {
      auto it = lgkmRemainingConsumers.find(lgkm);
      if (it == lgkmRemainingConsumers.end())
        continue; // LGKM not yet issued
      if (--it->second <= 0) {
        lgkmRemainingConsumers.erase(it);
        outstandingLgkm = std::max(int64_t{0}, outstandingLgkm - 1);
      }
    }
    break;
  }
  case QueueType::LGKM:
    outstandingLgkm++;
    if (auto it = lgkmTotalConsumers.find(chosen);
        it != lgkmTotalConsumers.end())
      lgkmRemainingConsumers[chosen] = it->second;
    break;
  case QueueType::VMEM:
    outstandingVmem++;
    break;
  default:
    break;
  }
}

void SchedState::classifyOps(const SchedGraph &schedGraph,
                             const CDNA3Latencies &L) {
  int32_t nNodes = schedGraph.sizeNodes();
  queueTypes.resize(nNodes);
  execLatencies.resize(nNodes);
  issueCosts.resize(nNodes);
  waitKind.assign(nNodes, WK_NotWait);
  for (auto [i, op] : llvm::enumerate(schedGraph.getOps())) {
    queueTypes[i] = classifyOp(op);
    execLatencies[i] = getExecLatency(op, queueTypes[i], L);
    issueCosts[i] = getIssueCost(op, queueTypes[i], L);
    if (auto waitOp = dyn_cast<WaitOp>(op)) {
      bool wvm = waitOp.getVmCnt() != WaitOp::kNoWaitCount;
      bool wlgkm = waitOp.getLgkmCnt() != WaitOp::kNoWaitCount;
      for (Value dep : waitOp.getDependencies()) {
        if (isFlatToken(dep.getType()))
          wvm = true;
        if (isLgkmToken(dep.getType()))
          wlgkm = true;
      }
      waitKind[i] = (wvm ? WK_VM : 0) | (wlgkm ? WK_LGKM : 0);
    }
  }

  for (int32_t nodeId = 0; nodeId < nNodes; ++nodeId) {
    if (queueTypes[nodeId] != QueueType::LGKM)
      continue;
    int32_t totalConsumers = 0;
    for (const auto &edge : schedGraph.edges(nodeId)) {
      int32_t succId = edge.second;
      assert(succId >= 0 && succId < nNodes && "edge successor out of range");
      if (queueTypes[succId] == QueueType::XDL) {
        xdlLgkmPreds[succId].push_back(nodeId);
        totalConsumers++;
      }
    }
    if (totalConsumers > 0)
      lgkmTotalConsumers[nodeId] = totalConsumers;
  }
}

//===----------------------------------------------------------------------===//
// Barrier bypass mechanism
//===----------------------------------------------------------------------===//
namespace {
/// Per-barrier predecessor-closure from SchedGraph edges.
struct BarrierClosures {
  /// Factory.
  static BarrierClosures create(const SchedGraph &schedGraph);

  /// Update the barrier-closure tracking after `nodeId` is scheduled.
  void markIssued(const SchedGraph &schedGraph, int32_t nodeId);

  /// Returns the smallest node id of any unscheduled barrier.
  int32_t getNextUnscheduledBarrier() const;

  /// For each barrier node, the nodes that must execute before the barrier.
  DenseMap<int32_t, DenseSet<int32_t>> closure;

  /// For each barrier node, the count of memory ops in its closure that
  /// have not yet been issued.
  DenseMap<int32_t, int32_t> unscheduledMemPreds;

  /// For each node id, the list of barriers whose closure contains it.
  SmallVector<SmallVector<int32_t>> containingBarriers;
};
} // namespace

static bool isMemoryNode(const SchedGraph &schedGraph, int32_t n) {
  Operation *op = schedGraph.getOp(n);
  if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
    return instOp.hasAnyProps({InstProp::Ds, InstProp::IsVmem, InstProp::Smem});
  return false;
}

BarrierClosures BarrierClosures::create(const SchedGraph &schedGraph) {
  BarrierClosures result;
  result.containingBarriers.resize(schedGraph.sizeNodes());

  for (int32_t b = 0; b < schedGraph.sizeNodes(); ++b) {
    if (!isa<SBarrier>(schedGraph.getOp(b)))
      continue;
    DenseSet<int32_t> closure = schedGraph.bfs(b, /*reversrOrder=*/true);
    int32_t memCount = 0;
    for (int32_t n : closure) {
      result.containingBarriers[n].push_back(b);
      if (isMemoryNode(schedGraph, n))
        memCount++;
    }
    result.unscheduledMemPreds[b] = memCount;
    result.closure[b] = std::move(closure);
  }
  return result;
}

void BarrierClosures::markIssued(const SchedGraph &schedGraph, int32_t nodeId) {
  if (isa<SBarrier>(schedGraph.getOp(nodeId))) {
    unscheduledMemPreds.erase(nodeId);
  } else if (isMemoryNode(schedGraph, nodeId)) {
    for (int32_t b : containingBarriers[nodeId]) {
      auto it = unscheduledMemPreds.find(b);
      if (it != unscheduledMemPreds.end() && it->second > 0)
        it->second--;
    }
  }
}

int32_t BarrierClosures::getNextUnscheduledBarrier() const {
  int32_t result = -1;
  for (const auto &kv : unscheduledMemPreds) {
    if (result < 0 || kv.first < result)
      result = kv.first;
  }
  return result;
}

/// Pure scoring function: priority-ordered alternation with a per-occurrence
/// burst penalty.
static int computeScore(QueueType qt, int64_t stall, const SchedState &s,
                        bool consumerInReady, int32_t waitKind,
                        bool isVmemStore, bool isVmemStoreFeeder,
                        bool isXdlChainContinuation, const CDNA3Latencies &L) {
  // Dominance (no burst / saturation): store(~1750) > load(800) >
  // non-chain XDL(400) > feeder(250) > chain-cont XDL(<=300, often <0).
  int score = 0;

  // 1. Stall avoidance: stallWeight per cycle, capped at stallCap.
  if (stall > 0) {
    int64_t capped = std::min(stall, L.stallCap);
    score -= L.stallWeight * static_cast<int>(capped);
  }

  // 2. Priority bonus by kind.
  switch (qt) {
  case QueueType::VMEM:
    score += L.vmemPriority;
    break;
  case QueueType::XDL:
    score += L.xdlPriority;
    break;
  case QueueType::LGKM:
    score += L.lgkmPriority;
    break;
  default:
    score += L.otherPriority;
    break;
  }

  // 3. Per-kind burst penalty: -perOccPenalty * count_in_lookback_window.
  // Skipped for stores and for VALU/SALU/Unknown.
  ArrayRef<QueueType> recent(s.recentKinds);
  size_t lookback = 0;
  int perOccPenalty = 0;
  switch (qt) {
  case QueueType::VMEM:
    if (isVmemStore)
      break;
    lookback = L.vmemBurstLookback;
    perOccPenalty = L.vmemBurstPenalty;
    break;
  case QueueType::LGKM:
    lookback = L.lgkmBurstLookback;
    perOccPenalty = L.lgkmBurstPenalty;
    break;
  case QueueType::XDL:
    lookback = L.xdlBurstLookback;
    perOccPenalty = L.xdlBurstPenalty;
    break;
  default:
    break;
  }
  if (lookback > 0) {
    ArrayRef<QueueType> window = recent.take_back(lookback);
    int recentCount = static_cast<int>(llvm::count(window, qt));
    score -= perOccPenalty * recentCount;
  }

  // 3a. VALU/SALU <-> LGKM/VMEM adjacency requires an 8-cycle NOP.
  QueueType prevKind = recent.empty() ? QueueType::Unknown : recent.back();
  if ((qt == QueueType::VALU &&
       (prevKind == QueueType::LGKM || prevKind == QueueType::VMEM)) ||
      ((qt == QueueType::LGKM || qt == QueueType::VMEM) &&
       prevKind == QueueType::VALU))
    score -= L.adjacencyPenalty;
  if ((qt == QueueType::SALU &&
       (prevKind == QueueType::LGKM || prevKind == QueueType::VMEM)) ||
      ((qt == QueueType::LGKM || qt == QueueType::VMEM) &&
       prevKind == QueueType::SALU))
    score -= L.adjacencyPenalty;

  // 3b. VMEM-load density cap on the L1/L2 read path; stores skip it.
  if (qt == QueueType::VMEM && !isVmemStore) {
    int vmemInWindow = static_cast<int>(llvm::count(recent, QueueType::VMEM));
    if (vmemInWindow >= L.vmemInWindowLimit)
      score -= L.vmemDensityPenalty;
  }

  // 5. Fire XDL whose LGKM producer is waiting on it to drain.
  if (qt == QueueType::XDL && consumerInReady)
    score += L.consumerReadyBonus;

  // 6. Wait deferral: stall ~= max(0, latency - opsSince * hideTimePerOp).
  if (waitKind != 0 && qt == QueueType::Unknown) {
    auto opsSince = [&](QueueType q) -> int {
      for (size_t i = 0; i < recent.size(); ++i) {
        if (recent[recent.size() - 1 - i] == q)
          return static_cast<int>(i);
      }
      return static_cast<int>(recent.size());
    };
    if (waitKind & SchedState::WK_VM) {
      int hideDone = opsSince(QueueType::VMEM) * L.waitHideTimePerOp;
      int remainingStall = std::max(0, L.waitMemoryLatency - hideDone);
      score -= L.stallWeight * remainingStall;
    }
    if (waitKind & SchedState::WK_LGKM) {
      int hideDone = opsSince(QueueType::LGKM) * L.waitHideTimePerOp;
      int remainingStall = std::max(0, L.waitLgkmExecLatency - hideDone);
      score -= L.stallWeight * remainingStall;
    }
  }

  // 7. VMCNT 4-bit saturation cap.
  if (qt == QueueType::VMEM && s.outstandingVmem >= L.vmemSaturationThreshold)
    score -= L.vmemSaturationPenalty;

  // 8. VMEM store bonus (uncapped; saturation penalty above damps it).
  if (isVmemStore && qt == QueueType::VMEM)
    score += L.vmemStoreBonus;

  // 9. VMEM store-feeder bonus (any op transitively feeding a store).
  if (isVmemStoreFeeder)
    score += L.vmemStoreFeederBonus;

  // 10. XDL chain-continuation stall estimate (RAW on accumulator).
  if (isXdlChainContinuation)
    score -= L.xdlChainStallEstimate;

  return score;
}

/// Pick the index in `ready` with the highest `computeScore`. Ties broken by
/// smallest node id for a deterministic order. Returns -1 if `ready` is empty.
static int32_t pickByScore(ArrayRef<int32_t> ready, const QueueSimulator &sim,
                           ArrayRef<QueueType> queueTypes,
                           ArrayRef<int32_t> waitKind, const SchedState &state,
                           function_ref<bool(int32_t)> hasXdlConsumerInReady,
                           function_ref<bool(int32_t)> isVmemStoreOp,
                           function_ref<bool(int32_t)> isVmemStoreFeederOp,
                           function_ref<bool(int32_t)> isXdlChainContinuationOp,
                           const CDNA3Latencies &L) {
  int32_t bestIdx = -1;
  int bestScore = std::numeric_limits<int>::min();
  for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
    int32_t nodeId = ready[i];
    QueueType qt = queueTypes[nodeId];
    int64_t stall = sim.wouldStall(qt, L);
    bool consumerInReady =
        (qt == QueueType::XDL) && hasXdlConsumerInReady(nodeId);
    bool isVmemStore = (qt == QueueType::VMEM) && isVmemStoreOp(nodeId);
    bool isFeeder = !isVmemStore && isVmemStoreFeederOp(nodeId);
    bool isXdlChain =
        (qt == QueueType::XDL) && isXdlChainContinuationOp(nodeId);
    int score =
        computeScore(qt, stall, state, consumerInReady, waitKind[nodeId],
                     isVmemStore, isFeeder, isXdlChain, L);
    if (score > bestScore ||
        (score == bestScore && bestIdx >= 0 && nodeId < ready[bestIdx])) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/// Annotate each op in `sched` with `sched.stall_cycles` and (if non-zero)
/// `sched.stall_reason` attributes, computed by the greedy schedFn. The
/// schedule order maps 1-to-1 onto the per-op stall vectors.
static void annotateStalls(const SchedGraph &schedGraph,
                           ArrayRef<int32_t> sched,
                           ArrayRef<int64_t> stallCycles,
                           ArrayRef<StringRef> stallReasons) {
  assert(stallCycles.size() == sched.size() &&
         "sched and stallCycles must have the same size");
  assert(stallReasons.size() == sched.size() &&
         "sched and stallReasons must have the same size");

  MLIRContext *ctx = schedGraph.getOp(0)->getContext();
  auto stallAttr = StringAttr::get(ctx, "sched.stall_cycles");
  auto reasonAttr = StringAttr::get(ctx, "sched.stall_reason");
  auto i64Ty = IntegerType::get(ctx, 64);

  for (size_t i = 0, n = sched.size(); i < n; ++i) {
    Operation *op = schedGraph.getOp(sched[i]);
    int64_t stall = stallCycles[i];
    StringRef reason = stallReasons[i];
    op->setAttr(stallAttr, IntegerAttr::get(i64Ty, stall));
    if (stall > 0)
      op->setAttr(reasonAttr, StringAttr::get(ctx, (reason + " full").str()));
  }
}

/// For each XDL op, record the (single-hop) XDL op whose result is used as
/// one of its operands -- the previous link of an MFMA accumulator chain.
/// The chain-cont stall penalty applies when this predecessor was issued
/// within `xdlChainLookback` of recent ops.
static llvm::DenseMap<int32_t, int32_t>
computeXdlChainPredecessors(const SchedGraph &schedGraph,
                            ArrayRef<QueueType> queueTypes) {
  llvm::DenseMap<int32_t, int32_t> xdlChainPred;
  ArrayRef<Operation *> ops = schedGraph.getOps();
  for (int32_t i = 0, e = static_cast<int32_t>(ops.size()); i < e; ++i) {
    if (queueTypes[i] != QueueType::XDL)
      continue;
    for (Value operand : ops[i]->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      int32_t defIdx = static_cast<int32_t>(schedGraph.getOpId(def));
      if (defIdx >= 0 && queueTypes[defIdx] == QueueType::XDL) {
        xdlChainPred[i] = defIdx;
        break;
      }
    }
  }
  return xdlChainPred;
}

/// Set of address-arith ops that transitively feed a VMEM store. Uses
/// `getBackwardSlice` (mlir/Analysis/SliceAnalysis.h); the filter stops
/// the walk at XDL/VMEM (MFMAs / loads) so it stays on the address chain
/// and at the block boundary (via `schedGraph.getOpId(op) < 0`).
static llvm::DenseSet<int32_t>
computeStoreFeederSet(const SchedGraph &schedGraph,
                      ArrayRef<QueueType> queueTypes) {
  llvm::DenseSet<int32_t> storeFeederSet;
  BackwardSliceOptions opts;
  opts.omitBlockArguments = true;
  opts.inclusive = true;
  opts.filter = [&](Operation *op) {
    int64_t idx = schedGraph.getOpId(op);
    if (idx < 0)
      return false;
    QueueType qt = queueTypes[idx];
    return qt != QueueType::XDL && qt != QueueType::VMEM;
  };
  SetVector<Operation *> slice;
  for (Operation *op : schedGraph.getOps()) {
    if (!isa<GlobalStoreDwordInstOpInterface, BufferStoreInstOpInterface>(op))
      continue;
    for (Value operand : op->getOperands()) {
      slice.clear();
      (void)getBackwardSlice(operand, &slice, opts);
      for (Operation *def : slice)
        storeFeederSet.insert(static_cast<int32_t>(schedGraph.getOpId(def)));
    }
  }
  return storeFeederSet;
}

LogicalResult
LowLevelSchedulerAttr::createSched(const SchedGraph &schedGraph,
                                   SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  if (schedGraph.getOps().empty())
    return success();

  // Look up the magic-number table for this scheduling run.
  const CDNA3Latencies &L = latencies(getPreset());

  // Classify each op (queue, exec latency, issue cost, wait-kind bitmask),
  // and pre-cache the LGKM-R <-> XDL adjacency for the deadline counter.
  SchedState state;
  state.classifyOps(schedGraph, L);
  QueueSimulator sim;

  llvm::DenseMap<int32_t, int32_t> xdlChainPred =
      computeXdlChainPredecessors(schedGraph, state.queueTypes);
  llvm::DenseSet<int32_t> storeFeederSet =
      computeStoreFeederSet(schedGraph, state.queueTypes);

  // Barrier-aware structural scheduling.
  // Barrier bypass mode triggers when the next unscheduled barrier has no more
  // unscheduled memory ops in its closure. We then skip the score-based loop
  // barrier-first, then mfmas. This gives mfmas opportunities to hide the issue
  // of post-barrier memory ops.
  BarrierClosures barriers = BarrierClosures::create(schedGraph);

  // Track stall cycles and reasons for each scheduled op.
  SmallVector<int64_t> stallCycles;
  SmallVector<StringRef> stallReasons;

  // Greedy pick driven by SchedGraph::topologicalSched. When scores tie, prefer
  // the smallest node id for a stable order. Assigns indices to the scheduled
  // operations.
  auto schedFn = [&](ArrayRef<int32_t> ready, SetVector<int32_t> &indices) {
    int32_t bestIdx = -1;
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

    // Barrier-bypass mode: when the next unscheduled barrier has no more
    // unscheduled memory preds in its closure, force-pick the barrier (if
    // ready) so it fires ahead of the score-based candidates.
    int32_t nextBarrier = barriers.getNextUnscheduledBarrier();
    if (nextBarrier >= 0 && barriers.unscheduledMemPreds[nextBarrier] == 0) {
      for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
        if (ready[i] == nextBarrier) {
          bestIdx = i;
          break;
        }
      }
      if (bestIdx < 0) {
        const DenseSet<int32_t> &closure = barriers.closure[nextBarrier];
        for (int32_t i = 0; i < static_cast<int32_t>(ready.size()); ++i) {
          if (closure.contains(ready[i])) {
            bestIdx = i;
            break;
          }
        }
      }
    }

    // Scoring mode. The consumer-ready test is a SchedState method; bind
    // it here so pickByScore can call it without seeing SchedState.
    auto hasXdlConsumerInReady = [&](int32_t xdlNode) {
      return state.hasXdlConsumerInReady(xdlNode);
    };
    auto isVmemStoreOp = [&](int32_t node) {
      Operation *op = schedGraph.getOp(node);
      return isa<GlobalStoreDwordInstOpInterface, BufferStoreInstOpInterface>(
          op);
    };
    auto isVmemStoreFeederOp = [&](int32_t node) {
      return storeFeederSet.contains(node);
    };
    auto isXdlChainContinuationOp = [&](int32_t node) {
      auto it = xdlChainPred.find(node);
      if (it == xdlChainPred.end())
        return false;
      int32_t pred = it->second;
      // Recent-id history is on the SchedState; walk the last
      // `xdlChainLookback` entries.
      ArrayRef<int32_t> recentIds(state.recentIds);
      size_t lb = L.xdlChainLookback;
      ArrayRef<int32_t> window = recentIds.take_back(lb);
      return llvm::is_contained(window, pred);
    };
    if (bestIdx < 0)
      bestIdx = pickByScore(ready, sim, state.queueTypes, state.waitKind, state,
                            hasXdlConsumerInReady, isVmemStoreOp,
                            isVmemStoreFeederOp, isXdlChainContinuationOp, L);

    assert(bestIdx >= 0 && "schedFn must select a ready node");

    // Record stall cycles and reasons.
    int32_t chosen = ready[bestIdx];
    QueueType chosenQt = state.queueTypes[chosen];
    int64_t stall = sim.issue(chosenQt, state.execLatencies[chosen],
                              state.issueCosts[chosen], L);
    stallCycles.push_back(stall);
    stallReasons.push_back(stall > 0 ? getQueueName(chosenQt) : "");

    barriers.markIssued(schedGraph, chosen);

    // Update closure state. Push the chosen kind into the sliding window;
    // drop the oldest if it has overflowed.
    state.recentKinds.push_back(chosenQt);
    state.recentIds.push_back(chosen);
    if (state.recentKinds.size() >
        static_cast<size_t>(SchedState::kRecentKindsWindow)) {
      state.recentKinds.erase(state.recentKinds.begin());
      state.recentIds.erase(state.recentIds.begin());
    }

    state.onIssued(chosen);

    indices.insert(bestIdx);
  };

  // Topologically schedule the operations.
  if (failed(schedGraph.topologicalSched(schedFn, sched)))
    return failure();

  // Annotate stalls for debugging purposes.
  if (getAnnotateStalls())
    annotateStalls(schedGraph, sched, stallCycles, stallReasons);

  return success();
}
