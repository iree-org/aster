//===- GraphColoring.h - Graph coloring for register allocation ---*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_LIB_DIALECT_AMDGCN_TRANSFORMS_GRAPHCOLORING_H
#define ASTER_LIB_DIALECT_AMDGCN_TRANSFORMS_GRAPHCOLORING_H

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::aster::amdgcn {

/// Per-node constraint carrying only integer data. The caller
/// (RegisterColoring) extracts these values from RangeConstraint before calling
/// colorGraph, so that GraphColoring has no dependency on mlir::Value or
/// RangeConstraint.
///
/// numRegs == 1 means singleton (no range constraint).
/// numRegs > 1 means this is the range leader for a multi-register allocation.
/// numRegs == 0 marks a non-leader range position; run() skips these via the
/// visited set, which the leader's colorNode call populates.
struct NodeConstraint {
  int32_t numRegs = 1;
  int32_t alignment = 1;
};

/// Assign physical register types to every unallocated node in the graph.
///
/// graph       — the already-materialized quotient graph (or original
///               interference graph cast to Graph) built by the caller.
/// constraints — one NodeConstraint per node in graph; the caller populates
///               this from RangeConstraint or quotientGraph.constraints.
///               Entries with numRegs == 0 are non-leader range positions
///               and are skipped.
/// types       — one AMDGCNRegisterTypeInterface per node; on entry holds the
///               node's current (unallocated) type; on exit holds the
///               allocated type for colored nodes; unchanged for
///               already-allocated nodes. The caller must ensure no entry has
///               value semantics (verified before calling).
/// loc         — diagnostic location used when allocation fails (typically the
///               enclosing function's loc).
/// numVGPRs    — number of available VGPRs (default 256).
/// numAGPRs    — number of available AGPRs (default 256).
/// numSGPRs    — number of available SGPRs (default 102). AMD GCN architectures
///               impose a fixed limit of 102 SGPRs per wavefront.
///
/// Returns failure if any node cannot be allocated (register pressure
/// exceeded).
LogicalResult colorGraph(const aster::Graph &graph,
                         ArrayRef<NodeConstraint> constraints,
                         MutableArrayRef<AMDGCNRegisterTypeInterface> types,
                         Location loc, int32_t numVGPRs = 256,
                         int32_t numAGPRs = 256, int32_t numSGPRs = 102);

/// Objective for the ILP register allocator.
enum class ILPObjective {
  /// Minimize the peak physical register used per register kind.
  MinPressure,
  /// Only prove that a feasible coloring exists; do not optimize.
  Feasibility,
};

/// ILP-based register allocator (CP-SAT, OR-Tools). Same contract as
/// colorGraph: writes allocated types into types[i] for previously
/// unallocated nodes, leaves already-allocated and value-semantics nodes
/// untouched, and preserves the leader/non-leader convention encoded in
/// constraints. Returns failure on infeasibility (no spill insertion).
///
/// ilpTimeLimitMs: wall-clock time limit in milliseconds; 0 disables the limit.
LogicalResult
colorGraphILP(const aster::Graph &graph, ArrayRef<NodeConstraint> constraints,
              MutableArrayRef<AMDGCNRegisterTypeInterface> types, Location loc,
              int32_t numVGPRs = 256, int32_t numAGPRs = 256,
              int32_t numSGPRs = 102, int32_t ilpTimeLimitMs = 5000,
              ILPObjective ilpObjective = ILPObjective::MinPressure);

} // namespace mlir::aster::amdgcn

#endif // ASTER_LIB_DIALECT_AMDGCN_TRANSFORMS_GRAPHCOLORING_H
