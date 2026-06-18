//===- ILPSchedulerAttrs.cpp - ILP scheduler external model ---------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SchedBuilderAttrInterface external model that schedules a basic block with an
// ILP (CP-SAT) model. Kept in its own translation unit so the solver
// dependency stays isolated from the greedy scheduler.
//
// DENSE RANK (position) model. Decision variable: t[i] = issue rank of node i,
// a permutation of [0, n). The three levels ADD constraints on the SAME model:
//
//   variables    t[i] in [0, n)                                 issue rank
//   (a) order    AllDifferent(t)                                dense
//   permutation
//                t[u] + 1 <= t[v]    for (u,v) in E             graph
//                precedence
//   (b) spread   per memory queue Q: Cumulative_1{ [t[i], t[i]+gap_Q) }
//   [level>=1]
//                (same-kind VMEM/LGKM ops >= gap_Q ranks apart -> spread
//                evenly)
//   (c) hide     XDL: Cumulative_1{ [t[i], t[i]+gap_X) } [level>=2]
//                (MFMAs >= gap_X ranks apart)
//
//   minimize     sum_i (n - i) * t[i]            unique source-order tie-break
//
// Why a DENSE permutation (not issue cycles): the issue port is the bottleneck
// ("issue-bound"). In a dense rank permutation every rank is used, so forcing
// the MFMAs gap_X ranks apart LEAVES gap_X-1 other ops between them BY
// CONSTRUCTION -- "mfma + (gap_X-1)" interleaving, independent of the
// objective. A cycle/makespan model allows idle slots, so it is indifferent to
// whether the other ops sit between the MFMAs or after them, and cannot force
// interleaving.
//
// The gaps are the tunable cost: gap_X = mfma+x interleaving, gap_{VMEM,LGKM} =
// even load/ds spreading. They are NOT the greedy scheduler's queue depths (the
// cost model is tuned to make the GREEDY heuristic interleave); the ILP is
// principled and wants its own spacing, so the gaps are env-overridable for a
// sweep:  ASTER_ILP_MFMA_GAP / ASTER_ILP_VMEM_GAP / ASTER_ILP_LGKM_GAP.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Scheduler/SchedulerCostModel.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "ortools/sat/cp_model.h"
#include "ortools/util/sorted_interval_list.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace ors = operations_research::sat;
namespace or_ns = operations_research;

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

// ISA detection mirrors the greedy scheduler (default CDNA3 if no parent
// module). Only used to classify ops into queues (the gaps come from env).
static ISAVersion detectISA(const SchedGraph &schedGraph) {
  if (Operation *parent = schedGraph.getBlock()->getParentOp())
    if (auto moduleOp = parent->getParentOfType<amdgcn::ModuleOp>())
      return getIsaForTarget(moduleOp.getTarget());
  return ISAVersion::CDNA3;
}

struct ILPSchedulerAttrImpl
    : SchedBuilderAttrInterface::FallbackModel<ILPSchedulerAttrImpl> {
  LogicalResult createSched(Attribute attr, const SchedGraph &schedGraph,
                            SmallVectorImpl<int32_t> &sched) const {
    if (!schedGraph.isCompressed())
      return failure();
    int32_t n = schedGraph.sizeNodes();
    if (n == 0)
      return success();

    auto ilpAttr = cast<ILPSchedulerAttr>(attr);
    int32_t level = ilpAttr.getLevel();
    int32_t timeLimitMs = ilpAttr.getTimeLimitMs();

    CDNA3Latencies L = latencies(detectISA(schedGraph), 1);
    NodeCostInfo cost = classifyGraph(schedGraph.getOps(), L);

    // Per-queue rank spacing -- the tunable interleaving cost (pass options).
    // mfmaGap spreads MFMAs ("hide mfmaGap-1 ops behind each MFMA");
    // vmemGap/lgkmGap spread loads / ds evenly. <=1 disables.
    int64_t mfmaGap = ilpAttr.getMfmaGap();
    int64_t vmemGap = ilpAttr.getVmemGap();
    int64_t lgkmGap = ilpAttr.getLgkmGap();
    // Barrier bypass: the graph conservatively pins every DS/VMEM op to each
    // s_barrier (memory fence), which fences the next iteration's loads/ds out
    // of the current MFMA shadow. When the barrier is a cross-loop sync (the
    // data is already buffered) that pin is unnecessary -- dropping barrier-
    // incident edges lets those fillers cross into the shadow.
    bool barrierBypass = ilpAttr.getBarrierBypass();
    // Register-pressure bound: cap each memory load's live range so aggressive
    // interleaving cannot push loads so far ahead of their use that the live
    // value count exceeds the register budget. 0 = unbounded.
    int64_t maxLoadDistance = ilpAttr.getMaxLoadDistance();
    // LDS prefetch depth: force each ds_read to lead its consumer by >= this
    // many issue ranks so other ds_reads issue in the gap -- the consumer then
    // waits lgkmcnt(depth) instead of lgkmcnt(0), hiding the LDS read latency.
    int64_t minLgkmDistance = ilpAttr.getMinLgkmDistance();

    // Debug: dump the scheduling graph (nodes + queue class + precedence edges)
    // so multi-buffering / op independence is visible directly, without reading
    // post-coalesce asm. Enabled by ASTER_ILP_DUMP_GRAPH=1.
    if (std::getenv("ASTER_ILP_DUMP_GRAPH")) {
      ArrayRef<Operation *> ops = schedGraph.getOps();
      auto qname = [](QueueType q) -> const char * {
        switch (q) {
        case QueueType::VALU:
          return "VALU";
        case QueueType::XDL:
          return "XDL";
        case QueueType::SALU:
          return "SALU";
        case QueueType::VMEM:
          return "VMEM";
        case QueueType::LGKM:
          return "LGKM";
        default:
          return "OTHER";
        }
      };
      llvm::errs() << "=== ILP-SCHED-GRAPH n=" << n << " level=" << level
                   << " edges=" << schedGraph.getEdges().size() << " ===\n";
      for (int32_t i = 0; i < n; ++i)
        llvm::errs() << "n" << i << " [" << qname(cost.queueTypes[i]) << "] "
                     << ops[i]->getName().getStringRef() << "\n";
      for (const Graph::Edge &e : schedGraph.getEdges())
        llvm::errs() << "e " << e.first << " -> " << e.second << "\n";
      llvm::errs() << "=== END ILP-SCHED-GRAPH ===\n";
    }

    // Issue cost: free ops take NO issue slot -- they must neither pad the
    // schedule nor satisfy a same-kind gap with non-work. Real ops take one
    // slot. Free = (a) register-allocation/SSA setup (constants, alloca,
    // register-range), and (b) SYNC markers -- amdgcn.wait and s_barrier. A
    // wait is cost-aware: in steady state it gates a cross-loop
    // (already-buffered) value, so it does not stall and occupies no issue
    // slot; its SSA token operands keep it ordered after its producers, and
    // WaitAnalysis re-derives the real hardware wait counts on the final
    // schedule. Treating waits as free is what stops them from acting as fake
    // fillers between MFMAs (so the gap is filled by real loads/ds instead).
    ArrayRef<Operation *> nodeOps = schedGraph.getOps();
    auto isFreeOp = [](Operation *op) {
      StringRef nm = op->getName().getStringRef();
      return nm == "arith.constant" || nm == "amdgcn.alloca" ||
             nm == "amdgcn.make_register_range" ||
             nm == "amdgcn.split_register_range" || nm == "amdgcn.wait" ||
             nm == "amdgcn.s_barrier";
    };
    SmallVector<int64_t> issueCost(n, 1);
    for (int32_t i = 0; i < n; ++i)
      if (isFreeOp(nodeOps[i]))
        issueCost[i] = 0;

    ArrayRef<Graph::Edge> edges = schedGraph.getEdges();
    bool dump = std::getenv("ASTER_ILP_DUMP_GRAPH") != nullptr;

    // Build + solve the dense-rank issue model over `subset` (global node ids,
    // in block order) and append the solved order to `out`. Used once for the
    // whole block, or once per window. Only edges with BOTH endpoints in the
    // subset are constrained; cross-subset precedence is guaranteed by the
    // window assignment (producer-window <= consumer-window), so concatenating
    // subsets in window order respects all precedence.
    //
    // (a) issue port: real ops occupy [c, c+1) (no-overlap sequences them);
    //     free ops occupy [c, c) and never advance the real-work clock.
    // (pressure) maxLoadDistance bounds each load's live range.
    // (b, c) same-kind NoOverlap of [c, c+gap) forces gap-1 real ops between
    //     same-kind ops (mfma + (gap-1) interleaving, even load/ds spreading).
    // objective: minimize the real-op issue makespan (idle real-slot pushes
    //     work later) with a source-order tie-break; wMakespan = m^3+1 strictly
    //     dominates the tie-break (which can reach ~m^3/2).
    auto solveSubset = [&](ArrayRef<int32_t> subset, int32_t windowId,
                           SmallVectorImpl<int32_t> &out) -> LogicalResult {
      int32_t m = static_cast<int32_t>(subset.size());
      if (m == 0)
        return success();
      llvm::DenseMap<int32_t, int32_t> g2l;
      for (int32_t li = 0; li < m; ++li)
        g2l[subset[li]] = li;

      ors::CpModelBuilder model;
      // Rank horizon. The spacing can legitimately push the last rank past the
      // node count m: K same-kind ops at gap G need (K-1)*G+1 ranks, which
      // exceeds m when the window is filler-poor. It is ALWAYS feasible to
      // spread ops later, so the rank domain must allow that -- capping at m
      // would make a filler-poor window spuriously infeasible. m*maxGap holds
      // the worst case (every op maxGap apart, trivially gap- and order-
      // feasible); with no active spacing (maxGap==1) it is the dense [0, m]
      // model unchanged. The makespan objective still packs fillers into the
      // gaps (that is the lower-makespan placement), so interleaving is intact.
      int64_t maxGap = 1;
      if (level >= 1)
        maxGap = std::max(maxGap, std::max(vmemGap, lgkmGap));
      if (level >= 2)
        maxGap = std::max(maxGap, mfmaGap);
      // minLgkmDistance (prefetch depth) pushes a consumer up to that many
      // ranks past its ds_read, so the horizon must allow it (else spurious
      // infeasibility from the rank cap).
      int64_t H = static_cast<int64_t>(m) * maxGap + minLgkmDistance;
      std::vector<ors::IntVar> c;
      c.reserve(m);
      for (int32_t li = 0; li < m; ++li)
        c.push_back(model.NewIntVar(or_ns::Domain(0, H)));
      SmallVector<ors::IntervalVar> issueIv;
      issueIv.reserve(m);
      for (int32_t li = 0; li < m; ++li)
        issueIv.push_back(
            model.NewFixedSizeIntervalVar(c[li], issueCost[subset[li]]));
      model.AddNoOverlap(issueIv);

      for (const Graph::Edge &e : edges) {
        auto u = g2l.find(e.first), v = g2l.find(e.second);
        if (u == g2l.end() || v == g2l.end())
          continue;
        if (barrierBypass && (isa<amdgcn::SBarrier>(nodeOps[e.first]) ||
                              isa<amdgcn::SBarrier>(nodeOps[e.second])))
          continue;
        model.AddLessOrEqual(ors::LinearExpr(c[u->second]) + issueCost[e.first],
                             c[v->second]);
      }
      if (maxLoadDistance > 0)
        for (const Graph::Edge &e : edges) {
          auto u = g2l.find(e.first), v = g2l.find(e.second);
          if (u == g2l.end() || v == g2l.end())
            continue;
          QueueType pq = cost.queueTypes[e.first];
          if (pq != QueueType::VMEM && pq != QueueType::LGKM)
            continue;
          model.AddLessOrEqual(c[v->second],
                               ors::LinearExpr(c[u->second]) + maxLoadDistance);
        }
      if (minLgkmDistance > 0)
        for (const Graph::Edge &e : edges) {
          auto u = g2l.find(e.first), v = g2l.find(e.second);
          if (u == g2l.end() || v == g2l.end())
            continue;
          if (cost.queueTypes[e.first] != QueueType::LGKM)
            continue;
          model.AddLessOrEqual(ors::LinearExpr(c[u->second]) + minLgkmDistance,
                               c[v->second]);
        }
      // Cross-window prefetch residual: a ds_read placed in an earlier window
      // must still lead its consumers here by the remaining ranks, else every
      // window start collapses to lgkmcnt(0). Producers in `out` have fixed
      // positions, so the residual is a constant bound; only the tail (within
      // minLgkmDistance real ops of the boundary) contributes.
      if (minLgkmDistance > 0 && !out.empty()) {
        int64_t realCount = 0;
        llvm::DenseMap<int32_t, int64_t> tailPos;
        for (int32_t id : out)
          realCount += issueCost[id];
        int64_t seen = 0;
        for (int32_t id : out) {
          seen += issueCost[id];
          if (cost.queueTypes[id] == QueueType::LGKM &&
              seen + minLgkmDistance > realCount)
            tailPos[id] = seen;
        }
        if (!tailPos.empty())
          for (const Graph::Edge &e : edges) {
            auto v = g2l.find(e.second);
            auto p = tailPos.find(e.first);
            if (v == g2l.end() || p == tailPos.end())
              continue;
            model.AddGreaterOrEqual(c[v->second],
                                    p->second + minLgkmDistance - realCount);
          }
      }

      auto addSpacing = [&](QueueType kind, int64_t gap) {
        if (gap <= 1)
          return;
        SmallVector<ors::IntervalVar> ivs;
        for (int32_t li = 0; li < m; ++li)
          if (cost.queueTypes[subset[li]] == kind)
            ivs.push_back(model.NewFixedSizeIntervalVar(c[li], gap));
        if (ivs.size() > 1)
          model.AddNoOverlap(ivs);
      };
      if (level >= 1) {
        addSpacing(QueueType::VMEM, vmemGap);
        addSpacing(QueueType::LGKM, lgkmGap);
      }
      if (level >= 2)
        addSpacing(QueueType::XDL, mfmaGap);

      ors::IntVar makespan = model.NewIntVar(or_ns::Domain(0, H));
      SmallVector<ors::LinearExpr> ends;
      for (int32_t li = 0; li < m; ++li)
        if (issueCost[subset[li]] > 0)
          ends.push_back(ors::LinearExpr(c[li]));
      if (!ends.empty())
        model.AddMaxEquality(makespan, ends);
      ors::LinearExpr srcBias;
      for (int32_t li = 0; li < m; ++li)
        srcBias += ors::LinearExpr::Term(c[li], static_cast<int64_t>(m - li));
      // Must dominate the srcBias tie-break, whose max is now H * m(m+1)/2
      // (c can reach H, not m); H*m*m+1 > that for m > 1.
      int64_t wMakespan = H * m * m + 1;
      model.Minimize(ors::LinearExpr::Term(makespan, wMakespan) + srcBias);

      ors::SatParameters params;
      params.set_num_search_workers(1);
      params.set_random_seed(1);
      params.set_relative_gap_limit(0.05);
      if (timeLimitMs > 0)
        params.set_max_deterministic_time(timeLimitMs / 1000.0);
      ors::CpSolverResponse resp =
          ors::SolveWithParameters(model.Build(), params);
      if (dump)
        llvm::errs() << "ILP-WINDOW=" << windowId << " m=" << m
                     << " STATUS=" << static_cast<int>(resp.status())
                     << " (4=OPT 2=FEAS 3=INFEAS 0=UNKNOWN)\n";
      if (resp.status() == ors::CpSolverStatus::MODEL_INVALID)
        llvm::report_fatal_error("ILP schedule model is invalid -- bug in "
                                 "ILPScheduler");
      if (resp.status() != ors::CpSolverStatus::OPTIMAL &&
          resp.status() != ors::CpSolverStatus::FEASIBLE)
        return failure();
      SmallVector<int64_t> tv(m);
      for (int32_t li = 0; li < m; ++li)
        tv[li] = ors::SolutionIntegerValue(resp, c[li]);
      SmallVector<int32_t> local = llvm::to_vector(llvm::seq<int32_t>(0, m));
      llvm::sort(local, [&](int32_t a, int32_t b) {
        return tv[a] != tv[b] ? tv[a] < tv[b] : a < b;
      });
      for (int32_t li : local)
        out.push_back(subset[li]);
      return success();
    };

    // A candidate schedule is accepted only if it is a permutation of all n
    // nodes respecting every precedence edge. A windowing bug can then only
    // cost a fall-back to the greedy scheduler -- never a wrong schedule.
    auto isValidSchedule = [&](ArrayRef<int32_t> order) -> bool {
      if (static_cast<int32_t>(order.size()) != n)
        return false;
      SmallVector<int32_t> pos(n, -1);
      for (int32_t p = 0; p < n; ++p) {
        int32_t id = order[p];
        if (id < 0 || id >= n || pos[id] != -1)
          return false;
        pos[id] = p;
      }
      for (const Graph::Edge &e : edges)
        if (pos[e.first] >= pos[e.second])
          return false;
      return true;
    };

    // Dispatch. The expensive part is CP-SAT's presolve on a big whole-block
    // model, so skip it where it does not buy hot-loop interleaving:
    //   Lever 1 -- a block with NO MFMA (prologue / epilogue / setup) is not
    //   the
    //     compute hot path, so it is not worth a whole-block ILP; the greedy
    //     low-level scheduler orders it well enough.
    // Windowing is disabled: the windowed solve dumps consumer-less
    // loop-carried prefetch loads into the last window, clustering them at the
    // iteration tail. At the back edge the whole cluster is in flight at once
    // and the count-based wait cannot drain it to vmcnt(0) before the next
    // iteration's loads reuse the same registers -- a loop-carried VMEM WAR
    // that surfaces as an illegal access. The whole-block solve spaces those
    // loads with vmemGap and stays safe. An MFMA block is always solved
    // whole-block.
    auto greedy = [&]() -> LogicalResult {
      return cast<SchedBuilderAttrInterface>(
                 LowLevelSchedulerAttr::get(ilpAttr.getContext(), /*preset=*/1,
                                            /*annotateStalls=*/false))
          .createSched(schedGraph, sched);
    };

    int32_t xdlCount = 0;
    for (int32_t i = 0; i < n; ++i)
      if (cost.queueTypes[i] == QueueType::XDL)
        ++xdlCount;

    // Lever 1: no MFMAs -> not perf-critical -> greedy (skip CP-SAT presolve).
    if (xdlCount == 0) {
      if (dump)
        llvm::errs() << "ILP-GREEDY (no MFMA, n=" << n << ")\n";
      return greedy();
    }

    // Solve the whole block as one ILP.
    SmallVector<int32_t> orderWhole;
    SmallVector<int32_t> allNodes = llvm::to_vector(llvm::seq<int32_t>(0, n));
    if (succeeded(solveSubset(allNodes, /*windowId=*/0, orderWhole)) &&
        isValidSchedule(orderWhole)) {
      sched.assign(orderWhole.begin(), orderWhole.end());
      return success();
    }

    // Never fail a legal kernel: greedy low-level schedule.
    if (Operation *first = schedGraph.getOp(0))
      mlir::emitRemark(first->getLoc())
          << "ILP scheduler: no solution within time limit; falling back to "
             "the greedy low-level scheduler (preset 1)";
    return greedy();
  }
};
} // namespace

namespace mlir::aster::amdgcn {
void attachILPSchedulerModel(MLIRContext *ctx) {
  ILPSchedulerAttr::attachInterface<ILPSchedulerAttrImpl>(*ctx);
}
} // namespace mlir::aster::amdgcn
