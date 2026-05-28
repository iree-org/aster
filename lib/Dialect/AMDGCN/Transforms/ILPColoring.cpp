//===- ILPColoring.cpp - CP-SAT ILP register allocator back-end ----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphColoring.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Support/Graph.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

#define DEBUG_TYPE "amdgcn-register-coloring-ilp"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
namespace ors = operations_research::sat;
// Domain lives in the outer operations_research namespace.
namespace or_ns = operations_research;

namespace {

/// Descriptor for a leader node in the ILP model. A leader is an unallocated
/// node that is responsible for allocating one or more contiguous registers.
struct LeaderVar {
  int32_t nodeId;
  int32_t numRegs;
  int32_t alignment;
  RegisterKind kind;
  int32_t maxRegs;
  ors::IntVar start;
  ors::IntervalVar interval;
};

/// Encapsulates the CP-SAT model construction and solution extraction for ILP
/// register allocation. Mirrors the interface contract of colorGraph.
struct ILPColoringImpl {
  ILPColoringImpl(const aster::Graph &graph,
                  ArrayRef<NodeConstraint> nodeConstraints,
                  MutableArrayRef<AMDGCNRegisterTypeInterface> types,
                  Location loc, int32_t numVGPRs, int32_t numAGPRs,
                  int32_t numSGPRs, int32_t ilpTimeLimitMs,
                  ILPObjective ilpObjective)
      : graph(graph), nodeConstraints(nodeConstraints), types(types), loc(loc),
        numVGPRs(numVGPRs), numAGPRs(numAGPRs), numSGPRs(numSGPRs),
        ilpTimeLimitMs(ilpTimeLimitMs), ilpObjective(ilpObjective) {}

  LogicalResult run();

private:
  /// Return the max register count for a given kind.
  int32_t maxRegsForKind(RegisterKind kind) const;

  /// Build LeaderVar entries for every unallocated leader node.
  void buildLeaders();

  /// Add alignment constraints: start == 0 (mod alignment).
  void addAlignmentConstraints();

  /// Add no-overlap constraints for interfering leaders of the same kind.
  void addInterferenceConstraints();

  /// Forbid each leader position from colliding with already-allocated
  /// neighbors of the same kind.
  void addFixedNeighborForbids();

  /// Add the min-pressure objective (minimize sum of per-kind peaks).
  void addPressureObjective();

  /// Extract the physical register assignments from the solver response and
  /// write them back into types.
  LogicalResult extractSolution(const ors::CpSolverResponse &resp);

  const aster::Graph &graph;
  ArrayRef<NodeConstraint> nodeConstraints;
  MutableArrayRef<AMDGCNRegisterTypeInterface> types;
  Location loc;
  int32_t numVGPRs;
  int32_t numAGPRs;
  int32_t numSGPRs;
  int32_t ilpTimeLimitMs;
  ILPObjective ilpObjective;

  ors::CpModelBuilder model;
  SmallVector<LeaderVar> leaders;
};

} // namespace

//===----------------------------------------------------------------------===//
// ILPColoringImpl implementation
//===----------------------------------------------------------------------===//

int32_t ILPColoringImpl::maxRegsForKind(RegisterKind kind) const {
  switch (kind) {
  case RegisterKind::SGPR:
    return numSGPRs;
  case RegisterKind::VGPR:
    return numVGPRs;
  case RegisterKind::AGPR:
    return numAGPRs;
  default:
    return 1;
  }
}

void ILPColoringImpl::buildLeaders() {
  int32_t n = static_cast<int32_t>(types.size());
  for (int32_t i = 0; i < n; ++i) {
    NodeConstraint nc = nodeConstraints[i];
    // Non-leader range positions are handled by their leader.
    if (nc.numRegs == 0)
      continue;
    AMDGCNRegisterTypeInterface regTy =
        dyn_cast<AMDGCNRegisterTypeInterface>(types[i]);
    if (!regTy)
      continue;
    // Skip already-allocated and value-semantics nodes.
    if (regTy.hasAllocatedSemantics() || regTy.hasValueSemantics())
      continue;

    RegisterKind kind = regTy.getRegisterKind();
    int32_t numRegs = nc.numRegs;
    int32_t alignment = nc.alignment;
    int32_t maxRegs = maxRegsForKind(kind);

    // Domain: [0, maxRegs - numRegs].
    ors::IntVar startVar =
        model.NewIntVar(or_ns::Domain(0, std::max(0, maxRegs - numRegs)));
    ors::IntervalVar intervalVar =
        model.NewFixedSizeIntervalVar(startVar, numRegs);

    leaders.push_back(
        {i, numRegs, alignment, kind, maxRegs, startVar, intervalVar});
  }
}

void ILPColoringImpl::addAlignmentConstraints() {
  for (const LeaderVar &lv : leaders) {
    if (lv.alignment <= 1)
      continue;
    // Enforce start == alignment * q by introducing quotient variable.
    // The tight upper bound is (maxRegs - numRegs) / alignment because start
    // is already bounded to [0, maxRegs - numRegs].
    ors::IntVar q = model.NewIntVar(or_ns::Domain(
        0, std::max(0, (lv.maxRegs - lv.numRegs) / lv.alignment)));
    model.AddEquality(lv.start, ors::LinearExpr::Term(q, lv.alignment));
  }
}

void ILPColoringImpl::addInterferenceConstraints() {
  int32_t numLeaders = static_cast<int32_t>(leaders.size());
  for (int32_t a = 0; a < numLeaders; ++a) {
    for (int32_t b = a + 1; b < numLeaders; ++b) {
      const LeaderVar &la = leaders[a];
      const LeaderVar &lb = leaders[b];
      // Only leaders of the same kind can interfere.
      if (la.kind != lb.kind)
        continue;
      // Check if any position of la has an edge to any position of lb.
      // graph.hasEdge is O(1) via DenseSet; the interference graph is
      // undirected so both directions are always present.
      bool interfere = false;
      for (int32_t pa = 0; pa < la.numRegs && !interfere; ++pa)
        for (int32_t pb = 0; pb < lb.numRegs && !interfere; ++pb)
          if (graph.hasEdge(la.nodeId + pa, lb.nodeId + pb))
            interfere = true;
      if (!interfere)
        continue;
      model.AddNoOverlap({la.interval, lb.interval});
    }
  }
}

void ILPColoringImpl::addFixedNeighborForbids() {
  for (const LeaderVar &lv : leaders) {
    for (int32_t pos = 0; pos < lv.numRegs; ++pos) {
      int32_t posId = lv.nodeId + pos;
      for (auto [src, tgt] : graph.edges(posId)) {
        AMDGCNRegisterTypeInterface regTy =
            dyn_cast<AMDGCNRegisterTypeInterface>(types[tgt]);
        if (!regTy)
          continue;
        if (!regTy.hasAllocatedSemantics())
          continue;
        if (regTy.getRegisterKind() != lv.kind)
          continue;
        assert(regTy.getAsRange().size() == 1 && "expected single register");
        int32_t physReg = regTy.getAsRange().begin().getRegister();
        // start + pos must not equal physReg, i.e. start != physReg - pos.
        int32_t forbidden = physReg - pos;
        if (forbidden >= 0 && forbidden <= lv.maxRegs - lv.numRegs)
          model.AddNotEqual(ors::LinearExpr(lv.start),
                            static_cast<int64_t>(forbidden));
      }
    }
  }
}

void ILPColoringImpl::addPressureObjective() {
  // For each register kind that has leaders, add a peak variable and minimize
  // the sum of peaks.
  llvm::DenseMap<int32_t, ors::IntVar> topByKind;
  for (const LeaderVar &lv : leaders) {
    int32_t kindVal = static_cast<int32_t>(lv.kind);
    if (topByKind.find(kindVal) == topByKind.end())
      topByKind[kindVal] =
          model.NewIntVar(or_ns::Domain(0, maxRegsForKind(lv.kind)));
    // top >= start + numRegs.
    model.AddGreaterOrEqual(topByKind[kindVal],
                            ors::LinearExpr(lv.start) + lv.numRegs);
  }

  SmallVector<ors::IntVar> tops;
  tops.reserve(topByKind.size());
  for (auto &[kindVal, topVar] : topByKind)
    tops.push_back(topVar);

  if (tops.empty())
    return;
  // Minimize sum of per-kind peaks.
  ors::LinearExpr objective;
  for (ors::IntVar topVar : tops)
    objective += topVar;
  model.Minimize(objective);
}

LogicalResult
ILPColoringImpl::extractSolution(const ors::CpSolverResponse &resp) {
  for (const LeaderVar &lv : leaders) {
    int64_t startVal = ors::SolutionIntegerValue(resp, lv.start);
    if (startVal < 0 || startVal + lv.numRegs > lv.maxRegs) {
      mlir::emitError(loc)
          << "ILP solver returned out-of-range register start — this is a bug "
             "in ILPColoring";
      return failure();
    }
    for (int32_t pos = 0; pos < lv.numRegs; ++pos) {
      int32_t nodeId = lv.nodeId + pos;
      Register reg = Register(static_cast<int16_t>(startVal + pos));
      types[nodeId] = cast<AMDGCNRegisterTypeInterface>(
          types[nodeId].cloneRegisterType(reg));
    }
  }
  return success();
}

LogicalResult ILPColoringImpl::run() {
  buildLeaders();
  addAlignmentConstraints();
  addInterferenceConstraints();
  addFixedNeighborForbids();

  if (ilpObjective == ILPObjective::MinPressure)
    addPressureObjective();
  // ILPObjective::Feasibility: no extra objective needed.

  LDBG() << "ILP register allocator: solving with " << leaders.size()
         << " leaders, time limit " << ilpTimeLimitMs << " ms";
  ors::SatParameters params;
  if (ilpTimeLimitMs > 0)
    params.set_max_time_in_seconds(ilpTimeLimitMs / 1000.0);
  ors::CpSolverResponse resp = ors::SolveWithParameters(model.Build(), params);

  if (resp.status() == ors::CpSolverStatus::OPTIMAL ||
      resp.status() == ors::CpSolverStatus::FEASIBLE)
    return extractSolution(resp);
  if (resp.status() == ors::CpSolverStatus::MODEL_INVALID)
    llvm::report_fatal_error(
        "ILP model is invalid — this is a bug in ILPColoring");
  if (resp.status() == ors::CpSolverStatus::UNKNOWN)
    // Time limit hit before a solution was found; anchor to the first leader.
    mlir::emitRemark(loc)
        << "ILP register allocator timed out; increase ilp-time-limit-ms";
  // INFEASIBLE or UNKNOWN: register pressure exceeded or time limit hit.
  return failure();
}

//===----------------------------------------------------------------------===//
// colorGraphILP entry point
//===----------------------------------------------------------------------===//

LogicalResult mlir::aster::amdgcn::colorGraphILP(
    const aster::Graph &graph, ArrayRef<NodeConstraint> constraints,
    MutableArrayRef<AMDGCNRegisterTypeInterface> types, Location loc,
    int32_t numVGPRs, int32_t numAGPRs, int32_t numSGPRs,
    int32_t ilpTimeLimitMs, ILPObjective ilpObjective) {
  return ILPColoringImpl(graph, constraints, types, loc, numVGPRs, numAGPRs,
                         numSGPRs, ilpTimeLimitMs, ilpObjective)
      .run();
}
