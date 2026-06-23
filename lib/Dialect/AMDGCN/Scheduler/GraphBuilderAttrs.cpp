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
#include "aster/Dialect/AMDGCN/Analysis/BufferAnalysis.h"
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
  void handleConservativeFenceBarriers(SchedGraph &graph);

  /// Add serialization edges for i1-producing ops within a block.
  void addI1SerializationEdges(SchedGraph &graph);

  /// Pin get_lds_offset and its offset users before dealloc_lds. SSA only
  /// connects alloc to dealloc on the buffer handle; offset consumers use a
  /// different value and are not otherwise ordered against deallocation.
  void buildLDSBufferLifetimeEdges(SchedGraph &graph);

  Block *block;
  DataFlowSolver &solver;
};
} // namespace

LogicalResult GraphBuilder::run(SchedGraph &graph) {
  buildSSADeps(graph);
  buildNonSSADeps(graph);
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

    // Add edges for the dependencies.
    ValueRange deps = op->getOperands();
    for (Value operand : deps) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getBlock() == block)
        graph.addEdge(producer, op);
    }
  }

  // Add serialization edges for i1-producing ops in the block, these are
  // additional edges that are not captured by SSA semantics only.
  addI1SerializationEdges(graph);
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

/// Add scheduling edges for read/write effects on fixed register resources:
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
  // Deal with conservative barriers first.
  handleConservativeFenceBarriers(graph);

  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    if (auto wait = dyn_cast<WaitCntOpInterface>(opIndex.value()))
      handleWaitOp(graph, opIndex.index(), wait);
  }

  // Add register effect edges for RAW / WAR / WAW edges for read/write effects
  // on architectural registers (SCC, VCC, M0) and preallocated physical
  // registers (VGPR/SGPR/AGPR).
  addRegisterEffectEdges(graph, block);

  // Add memory dependence edges for RAW / WAR / WAW).
  buildMemDepEdges(graph);

  /// Pin get_lds_offset its users before dealloc_lds (not tracked by SSA).
  buildLDSBufferLifetimeEdges(graph);
}

void GraphBuilder::buildLDSBufferLifetimeEdges(SchedGraph &graph) {
  auto pinBeforeDealloc = [&](Operation *user, DeallocLDSOp dealloc) {
    if (user->getBlock() == block)
      graph.addEdge(user, dealloc);
  };

  for (Operation &op : *block) {
    auto dealloc = dyn_cast<DeallocLDSOp>(&op);
    if (!dealloc)
      continue;

    Value buffer = dealloc.getBuffer();
    for (Operation &other : *block) {
      auto getOffset = dyn_cast<GetLDSOffsetOp>(&other);
      if (!getOffset || getOffset.getBuffer() != buffer)
        continue;

      // Mirror LDSInterferenceGraph::checkGetLDSOffset: the offset op and each
      // of its direct users must run while the buffer is still live.
      const auto *beforeGetOffset = solver.lookupState<BufferState>(
          solver.getProgramPointBefore(getOffset));
      if (beforeGetOffset &&
          beforeGetOffset->getBufferState(buffer) == BufferState::State::Dead)
        continue;

      pinBeforeDealloc(getOffset, dealloc);
      for (Operation *user : getOffset.getResult().getUsers())
        pinBeforeDealloc(user, dealloc);
    }
  }
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

void GraphBuilder::handleConservativeFenceBarriers(SchedGraph &graph) {
  // Helper: direction depends on relative position of op and barrier.
  auto addEdge = [&](std::pair<Operation *, int64_t> opWithPos,
                     std::pair<SBarrier, int64_t> barrierWithPos) {
    if (opWithPos.second < barrierWithPos.second)
      graph.addEdge(opWithPos.first, barrierWithPos.first);
    if (opWithPos.second > barrierWithPos.second)
      graph.addEdge(barrierWithPos.first, opWithPos.first);
  };

  // Barriers that require conservative fences.
  // TODO: ConservativeFenceOpInterface.
  SmallVector<std::pair<SBarrier, int64_t>> barriers;
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    auto barrier = dyn_cast<SBarrier>(opIndex.value());
    if (!barrier)
      continue;
    barriers.push_back({barrier, opIndex.index()});
  }

  // Iterate over all the operations in the graph and set conservative fences
  // for operations that have write effects visible by other threads / waves.
  for (auto opIndex : llvm::enumerate(graph.getOps())) {
    Operation *op = opIndex.value();
    int64_t i = opIndex.index();

    // All BarrierOpInterface, FenceOpInterface and FenceableOpInterface (with
    // an SSA fence token) fence against SBarrier.
    if (isa<BarrierOpInterface, FenceOpInterface>(op) ||
        (isa<FenceableOpInterface>(op) &&
         cast<FenceableOpInterface>(op).hasFenceToken())) {
      for (auto [barrier, barrierPos] : barriers)
        addEdge({op, i}, {barrier, barrierPos});
      continue;
    }

    // Ops with effects visible to other threads/waves (memory writes and VCC
    // writes), are not allowed to execute past the SBarrier immediately
    // following them.
    bool hasCrossThreadVisibleWrite = isa<StoreOpInterface>(op);
    if (!hasCrossThreadVisibleWrite) {
      if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
        llvm::DenseSet<Type> writeTypes;
        for (Value v : instOp.getInstOuts())
          appendRegTypesFromOperand(v, writeTypes);
        for (Type ty : writeTypes) {
          if (isa<VCCType, VCCLoType, VCCHiType>(ty)) {
            hasCrossThreadVisibleWrite = true;
            break;
          }
        }
      }
    }
    if (hasCrossThreadVisibleWrite) {
      for (auto [barrier, barrierPos] : barriers) {
        if (barrierPos > i) {
          graph.addEdge(op, barrier);
          break;
        }
      }
    }

    // Similarly, ops that depend on operations with effects visible to other
    // threads/waves (memory and VCC reads), are not allowed to execute prior to
    // the SBarrier immediately preceding them.
    bool needsFence = isa<LoadOpInterface>(op);
    if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
      llvm::DenseSet<Type> readTypes;
      for (Value v : instOp.getInstIns())
        appendRegTypesFromOperand(v, readTypes);
      for (Type ty : readTypes) {
        if (isa<VCCType, VCCLoType, VCCHiType>(ty)) {
          needsFence = true;
          break;
        }
      }
    }
    if (needsFence) {
      for (auto [barrier, barrierPos] : barriers) {
        if (barrierPos < i) {
          graph.addEdge(barrier, op);
          break;
        }
      }
    }
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
  DataFlowSolver &solver = analysis.getSolver();
  DominanceInfo &domInfo = analysis.getDomInfo();
  // Load the wait analysis so the graph builder can query wait states.
  loadWaitAnalysis(solver, domInfo, analysis.getIsaVersion());
  // Load buffer liveness for get_lds_offset lifetime edges.
  solver.load<BufferAnalysis>(domInfo);
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
