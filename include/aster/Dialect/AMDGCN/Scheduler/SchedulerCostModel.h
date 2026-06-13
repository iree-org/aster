//===- SchedulerCostModel.h - AMDGCN scheduler cost model -----------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-architecture latency / queue cost model shared by the AMDGCN instruction
// schedulers.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULERCOSTMODEL_H
#define ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULERCOSTMODEL_H

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::aster::amdgcn {

// Defined in AMDGCNInstOpInterface.h; only its name is needed here.
enum class ISAVersion : uint32_t;

// LGKM covers all LDS reads/writes and SMEM loads. Reads and writes have
// different raw latencies but in practice the lgkmcnt counter aggregates both,
// so distinguishing the two is not actionable (at least on CDNA3).
enum class QueueType : int32_t { VALU, XDL, SALU, VMEM, LGKM, Unknown };

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
  int64_t xdlExec16Pass = 64;   // 32x32x{1_2B,2,4_2B}
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
  // Hard cap on consecutive MFMA (deterministic spread dial; 0 = disabled).
  // When the trailing run of XDL ops already reaches xdlMaxRun, the next XDL is
  // penalized so any ready non-XDL op wins (forcing interleaving).
  int xdlMaxRun = 0;

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

/// Active per-architecture latency / policy table. `preset` selects a pre-tuned
/// magic-number set (1 = mfma-hiding default; future presets add a case).
CDNA3Latencies latencies(ISAVersion isa, int preset);

/// Classify an op into its hardware queue (`sched.queue` attr overrides).
QueueType classifyOp(Operation *op);

/// Exec latency in hw cycles (queue-slot hold). `sched.exec_latency` overrides.
int64_t getExecLatency(Operation *op, QueueType qt, const CDNA3Latencies &L);

/// Number of in-flight slots for a queue.
int64_t getQueueDepth(QueueType qt, const CDNA3Latencies &L);

/// Per-op issue cost in hw cycles -- how long the issue port is held.
int64_t getIssueCost(Operation *op, QueueType qt, const CDNA3Latencies &L);

/// Static per-node cost classification for a whole scheduling graph, indexed by
/// node id (ops[i] is node i). The batch form of classifyOp / getExecLatency /
/// getIssueCost; shared by the greedy and ILP schedulers.
struct NodeCostInfo {
  llvm::SmallVector<QueueType> queueTypes;
  llvm::SmallVector<int64_t> execLatencies;
  llvm::SmallVector<int64_t> issueCosts;
};

/// Classify every op in `ops` (node-id order) into queue / exec-latency /
/// issue-cost. Pure function of `ops` and `L`.
NodeCostInfo classifyGraph(llvm::ArrayRef<Operation *> ops,
                           const CDNA3Latencies &L);

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULERCOSTMODEL_H
