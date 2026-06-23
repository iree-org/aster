//===- GraphBuilderAttrs.cpp - AMDGCN graph builder attr impl -------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInterfaces.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Interfaces/DependentOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-sched"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// GraphBuilder - builds the SSA + non-SSA scheduling graph for a block
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

  /// Add hard ordering edges from the memory-dependence analysis, scoped to
  /// LDS. Same-buffer LDS RAW/WAR/WAW is a correctness constraint the
  /// post-schedule wait insertion cannot repair (DS ops issue in order, no hw
  /// interlock). Global ordering is left to the existing token/wait machinery.
  void buildMemDepEdges(SchedGraph &graph);

  /// Handle a wait operation.
  void handleWaitOp(SchedGraph &graph, int64_t pos, WaitCntOpInterface wait);

  /// Conservative pinning for hardware s_barrier.
  void handleSBarrier(SchedGraph &graph, int64_t pos, SBarrier barrier);

  /// Serialize barrier ops in program order.
  void addBarrierSerializationEdges(SchedGraph &graph);

  /// Add serialization edges for i1-producing ops within a block.
  void addI1SerializationEdges(SchedGraph &graph);

  Block *block;
  SmallVector<int64_t> syncPoints;
  DataFlowSolver &solver;
};
} // namespace

LogicalResult GraphBuilder::run(SchedGraph &graph) {
  buildSSADeps(graph);
  buildNonSSADeps(graph);
  buildMemDepEdges(graph);
  addI1SerializationEdges(graph);
  return success();
}

// Build the memory dependence edges for the graph.
void GraphBuilder::buildMemDepEdges(SchedGraph &graph) {
  auto mdaOr = MemoryDependenceAnalysis::create(
      block->getParentOp(),
      {GlobalMemoryResource::get(), LDSMemoryResource::get()});
  if (failed(mdaOr)) {
    LDBG() << "memdep analysis unavailable (non-flat CFG); skipping LDS edges";
    return;
  }
  MemoryDependenceAnalysis &mda = *mdaOr;
  for (Operation *op : graph.getOps())
    for (const MemDepEdge &edge : mda.getDependences(op))
      if (edge.resource == LDSMemoryResource::get() &&
          edge.producer->getBlock() == block &&
          edge.producer->isBeforeInBlock(op))
        graph.addEdge(edge.producer, op);
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
    // Exclude SOP1 mov ops and SOP2 SALU arithmetic ops (s_add_*, ...) from
    // sync points. The implicit read/write effects on architectural registers
    // SCC are captured by the register effect edges added in buildNonSSADeps.
    // Treating as fences would serialize independent arithmetic chains.
    bool isReorderableArith = false;
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op))
      isReorderableArith = instOp.hasAnyProps(
          {InstProp::Sop1, InstProp::Sop2, InstProp::IsValu});
    if ((!hasEffects || !mlir::isPure(op)) && !isReorderableArith &&
        !isa<LoadOpInterface, StoreOpInterface, AllocaOpInterface>(op)) {
      LDBG() << "Adding sync point: " << i
             << " op: " << OpWithFlags(op, OpPrintingFlags().skipRegions());
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

static bool archRegTypesOverlap(Type a, Type b) {
  if (isa<SCCType>(a))
    return isa<SCCType>(b);
  if (isa<SCCType>(b))
    return false;
  if (isa<M0Type>(a))
    return isa<M0Type>(b);
  if (isa<M0Type>(b))
    return false;
  // 0, 1, 2, 3 -> none, lo, hi, full.
  int ma = isa<VCCType>(a)     ? 3
           : isa<VCCLoType>(a) ? 1
           : isa<VCCHiType>(a) ? 2
                               : 0;
  int mb = isa<VCCType>(b)     ? 3
           : isa<VCCLoType>(b) ? 1
           : isa<VCCHiType>(b) ? 2
                               : 0;
  return ma && mb && (ma & mb);
}

/// True when two register resource types alias (arch half intersection or
/// allocated physical range overlap). Architectural and physical never overlap.
static bool regTypesOverlap(Type a, Type b) {
  if (archRegTypesOverlap(a, b))
    return true;
  auto regA = dyn_cast<AMDGCNRegisterTypeInterface>(a);
  auto regB = dyn_cast<AMDGCNRegisterTypeInterface>(b);
  if (!regA || !regB)
    return false;
  return regA.overlaps(regB);
}

static void appendRegTypesFromOperand(Value v, llvm::DenseSet<Type> &types) {
  auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
  if (!regTy || !regTy.hasAllocatedSemantics())
    return;
  SmallVector<RegisterTypeInterface> regs;
  if (failed(regTy.getUnderlyingRegisterTypes(regs)))
    return;
  for (RegisterTypeInterface r : regs)
    types.insert(r);
}

struct RegResourceState {
  SmallVector<std::pair<Operation *, Type>> writers;
  SmallVector<std::pair<Operation *, Type>> readers;
};

/// Add scheduling edges for read/write effects on register resources:
///   - architectural registers (SCC, VCC and their lo/hi halves, M0)
///   - preallocated physical registers (VGPR / SGPR / AGPR and ranges thereof)
// TODO: limit to last/observable writes
static void addRegisterEffectEdges(SchedGraph &graph, Block *block) {
  RegResourceState state;
  for (Operation &op : *block) {
    auto instOp = dyn_cast<AMDGCNInstOpInterface>(&op);
    if (!instOp)
      continue;

    llvm::DenseSet<Type> readTypes, writeTypes;
    for (Value v : instOp.getInstOuts())
      appendRegTypesFromOperand(v, writeTypes);
    for (Value v : instOp.getInstIns())
      appendRegTypesFromOperand(v, readTypes);
    if (readTypes.empty() && writeTypes.empty())
      continue;

    llvm::DenseSet<Type> allTypes = readTypes;
    allTypes.insert(writeTypes.begin(), writeTypes.end());
    for (Type ty : allTypes) {
      bool reads = readTypes.contains(ty);
      bool writes = writeTypes.contains(ty);
      if (reads) {
        for (auto [w, wt] : state.writers)
          if (w != &op && regTypesOverlap(ty, wt))
            graph.addEdge(w, &op);
      }
      if (writes) {
        for (auto [r, rt] : state.readers)
          if (r != &op && regTypesOverlap(ty, rt))
            graph.addEdge(r, &op);
        for (auto [w, wt] : state.writers)
          if (w != &op && regTypesOverlap(ty, wt))
            graph.addEdge(w, &op);
      }
      if (reads)
        state.readers.emplace_back(&op, ty);
      if (writes)
        state.writers.emplace_back(&op, ty);
    }
  }
}

void GraphBuilder::buildNonSSADeps(SchedGraph &graph) {
  // Preprocess and erase synchronization points.
  ArrayRef<Operation *> ops = graph.getOps();
  for (int64_t &i : syncPoints) {
    Operation *op = ops[i];
    if (auto waitOp = dyn_cast<WaitCntOpInterface>(op)) {
      handleWaitOp(graph, i, waitOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
    if (auto barrierOp = dyn_cast<SBarrier>(op)) {
      handleSBarrier(graph, i, barrierOp);
      // Mark the sync point as processed.
      i = -1;
      continue;
    }
    // Cross-wave ordering is entirely SSA, nothing to add here.
    if (isa<CrossWaveTokenBarrierOp>(op)) {
      i = -1;
      continue;
    }
  }

  // Erase all the processed sync points.
  llvm::erase(syncPoints, -1);

  addBarrierSerializationEdges(graph);

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

  // RAW / WAR / WAW edges for read/write effects on architectural registers
  // (SCC, VCC, M0) and preallocated physical registers (VGPR/SGPR/AGPR).
  addRegisterEffectEdges(graph, block);
}

static bool isTokenOfKind(Type type, MemoryInstructionKind kind) {
  if (auto rt = dyn_cast<ReadTokenType>(type))
    return rt.getKind() == kind;
  if (auto wt = dyn_cast<WriteTokenType>(type))
    return wt.getKind() == kind;
  return false;
}

static bool isLgkmToken(Type type) {
  return isTokenOfKind(type, MemoryInstructionKind::Shared) ||
         isTokenOfKind(type, MemoryInstructionKind::Constant);
}

// gfx1250 tensor tokens share the memory-latency class with flat tokens.
static bool isMemToken(Type type) {
  return isTokenOfKind(type, MemoryInstructionKind::Flat) ||
         isTokenOfKind(type, MemoryInstructionKind::Tensor);
}

void GraphBuilder::handleWaitOp(SchedGraph &graph, int64_t pos,
                                WaitCntOpInterface wait) {
  Operation *waitOp = wait.getOperation();
  // Get the wait state.
  const WaitState *state =
      solver.lookupState<WaitState>(solver.getProgramPointAfter(waitOp));
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
    graph.addEdge(op, waitOp);

  // Add edges for the operations that are after this wait operation that wait
  // on the same tokens.
  // Waits with explicit counters have side-effects and must be pinned.
  // Pin all DS ops, do not pin global / flat / constant ops (may be unsafe).
  // TODO: This is a crude approximation, needs proper futures + wait_all + then
  // modeling.
  // The flat-group counter is vmcnt (CDNA) or loadcnt/tensorcnt (gfx1250); the
  // lgkm-group is lgkmcnt (CDNA) or dscnt (gfx1250).
  bool hasExplicitVmCnt = wait.hasCounter(WaitCounterKind::Vm) ||
                          wait.hasCounter(WaitCounterKind::Load) ||
                          wait.hasCounter(WaitCounterKind::Tensor);
  bool hasExplicitLgkmCnt = wait.hasCounter(WaitCounterKind::Lgkm) ||
                            wait.hasCounter(WaitCounterKind::Ds);
  for (Value dep : waitOp->getOperands()) {
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
        graph.addEdge(waitOp, op);
    }

    bool producesVM = llvm::any_of(op->getResultTypes(), isMemToken);
    bool producesLgkm = llvm::any_of(op->getResultTypes(), isLgkmToken);
    // TODO: This is a crude approximation, needs proper futures + wait_all +
    // then modeling.
    if (hasExplicitVmCnt && producesVM)
      graph.addEdge(waitOp, op);
    if (hasExplicitLgkmCnt && producesLgkm)
      graph.addEdge(waitOp, op);
  }
}

void GraphBuilder::handleSBarrier(SchedGraph &graph, int64_t pos,
                                  SBarrier barrier) {
  // Direction depends on whether op is a predecessor or successor of the
  // barrier.
  auto addEdge = [&](Operation *op, int64_t i) {
    if (i < pos)
      graph.addEdge(op, barrier);
    if (i > pos)
      graph.addEdge(barrier, op);
  };

  // TODO: Gradually phase off these heuristics as we encourage finer-grained
  // barrier usage.

  // Iterate over all the operations in the graph.
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    // Skip self.
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

void GraphBuilder::addBarrierSerializationEdges(SchedGraph &graph) {
  Operation *prevBarrier = nullptr;
  for (Operation &op : *block) {
    if (!isa<BarrierOpInterface>(op))
      continue;
    if (prevBarrier)
      graph.addEdge(prevBarrier, &op);
    prevBarrier = &op;
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

//===----------------------------------------------------------------------===//
// Free functions used by SchedulerAttrs.cpp to implement ValueSchedulerAttr
//===----------------------------------------------------------------------===//

namespace mlir::aster::amdgcn {

LogicalResult initValueSchedulerAnalyses(SchedAnalysis &analysis) {
  // Load the wait analysis so the graph builder can query wait states.
  loadWaitAnalysis(analysis.getSolver(), analysis.getDomInfo(),
                   analysis.getIsaVersion());
  analysis.setRunDataflowAnalyses();
  return success();
}

FailureOr<SchedGraph> buildValueSchedulerGraph(Block *block,
                                               const SchedAnalysis &analysis) {
  SchedGraph graph(block);
  graph.setIsaVersion(analysis.getIsaVersion());
  GraphBuilder builder(block, analysis.getSolver());
  if (failed(builder.run(graph)))
    return failure();
  graph.compress();
  return graph;
}

} // namespace mlir::aster::amdgcn
