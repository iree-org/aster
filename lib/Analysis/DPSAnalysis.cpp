//===- DPSAnalysis.cpp - DPS analysis -------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/CFG.h"
#include "aster/IR/InstImpl.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/InterleavedRange.h"

using namespace mlir;
using namespace mlir::aster;

namespace {
using Allocation = DPSAnalysis::Allocation;
using AllocView = DPSAnalysis::AllocView;

/// Implementation helper that tracks the DPS alias equivalence classes and
/// value provenance while traversing the control flow graph.
struct DPSAnalysisImpl : CFGWalker<DPSAnalysisImpl> {
  using ProvenanceSet = DPSAnalysis::ProvenanceSet;
  DPSAnalysisImpl(DPSAnalysis &analysis,
                  DenseMap<Value, ProvenanceSet> &valueProvenance)
      : analysis(analysis), valueProvenance(valueProvenance) {}

  LogicalResult run(FunctionOpInterface funcOp);
  LogicalResult visitOp(Operation *op);
  LogicalResult visitControlFlowEdge(const BranchPoint &branchPoint,
                                     const Successor &successor);
  DPSAnalysis &analysis;
  DenseMap<Value, ProvenanceSet> &valueProvenance;
};
} // namespace

LogicalResult DPSAnalysisImpl::visitOp(Operation *op) {
  // Handle alloca operations.
  if (auto allocaOp = dyn_cast<AllocaOpInterface>(op)) {
    analysis.getOrCreateAlloc(allocaOp.getAlloca());
    return success();
  }

  // Handle instruction operations.
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (auto [out, res] : TiedInstOutsRange(instOp)) {
      if (!res)
        continue;
      AllocView allocView = analysis.getAllocView(out);
      if (!allocView.alloc || failed(analysis.mapAlloc(res, allocView.alloc)))
        return failure();
    }
    return success();
  }

  // Handle make register range operations.
  if (auto makeRegisterRangeOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
    Allocation *alloc = analysis.getOrCreateRange(
        makeRegisterRangeOp.getResult(), makeRegisterRangeOp.getInputs());
    if (!alloc)
      return failure();
    return success();
  }

  // Handle split register range operations.
  if (auto splitRegisterRangeOp = dyn_cast<amdgcn::SplitRegisterRangeOp>(op)) {
    AllocView allocView =
        analysis.getAllocView(splitRegisterRangeOp.getInput());
    if (!allocView.alloc ||
        failed(analysis.mapAlloc(splitRegisterRangeOp.getResults(),
                                 allocView.alloc)))
      return failure();
    return success();
  }
  return success();
}

LogicalResult
DPSAnalysisImpl::visitControlFlowEdge(const BranchPoint &branchPoint,
                                      const Successor &successor) {
  // Get the successor variables and try to create allocations for them.
  ValueRange succVars = successor.getInputs();
  for (Value value : succVars)
    (void)analysis.getOrCreateAlloc(value);

  // If the branch point is the entry point, return success.
  if (branchPoint.isEntryPoint())
    return success();

  // Get the branch point operands.
  ValueRange cfOperands = branchPoint.getOperands();
  assert(static_cast<int64_t>(succVars.size()) ==
             branchPoint.getNumOperands() &&
         "expected number of operands to match");

  // Drop the produced operands from the successor variables.
  succVars = succVars.drop_front(branchPoint.getProducedOperandCount());

  // Set the value provenance for the successor variables. Only track registers.
  for (auto [value, variable] : llvm::zip_equal(cfOperands, succVars)) {
    if (auto regTy = dyn_cast<RegisterTypeInterface>(variable.getType());
        !regTy || !regTy.hasValueSemantics())
      continue;
    valueProvenance[variable].insert({branchPoint.getPoint(), value});
  }
  return success();
}

LogicalResult DPSAnalysisImpl::run(FunctionOpInterface funcOp) {
  if (failed(walk(funcOp)))
    return failure();
  return success();
}

FailureOr<DPSAnalysis> DPSAnalysis::create(FunctionOpInterface funcOp) {
  DPSAnalysis analysis;
  DPSAnalysisImpl impl(analysis, analysis.valueProvenance);
  if (failed(impl.run(funcOp)))
    return failure();
  return analysis;
}

Allocation *DPSAnalysis::getOrCreateAlloc(Value value) {
  auto regTy = dyn_cast<RegisterTypeInterface>(value.getType());
  if (!regTy || !regTy.hasValueSemantics())
    return nullptr;

  // Try to get the allocation for the value.
  AllocView &allocView = allocViews[value];
  if (allocView.alloc)
    return allocView.alloc;

  // Create a new allocation for the value.
  Allocation *alloc = allocAllocator.Allocate(1);
  int32_t numRegs = regTy.getAsRange().size();
  new (alloc) Allocation(
      value, llvm::to_vector<1>(llvm::seq<int32_t>(nextId, nextId + numRegs)));

  // Map the allocation IDs with the allocation.
  for (int32_t id : alloc->ids)
    idToAlloc[id] = alloc;
  nextId += numRegs;
  allocView = AllocView(alloc, alloc->ids);
  return alloc;
}

Allocation *DPSAnalysis::getOrCreateRange(Value value, ValueRange range) {
  // Try to get the allocation for the range.
  AllocView &allocView = allocViews[value];
  if (allocView.alloc)
    return allocView.alloc;

  // Collect the allocation IDs for the range.
  SmallVector<int32_t, 1> ids;
  ids.reserve(range.size());
  for (Value value : range) {
    AllocView allocView = getAllocView(value);
    // Fail if the value was not already in the allocation map.
    if (!allocView.alloc)
      return nullptr;
    llvm::append_range(ids, allocView.ids);
  }

  // Create a new allocation for the range. IDs in the range are already
  // registered in idToAlloc from their original allocations; no new IDs.
  Allocation *alloc = allocAllocator.Allocate(1);
  new (alloc) Allocation(value, std::move(ids));
  allocView = AllocView(alloc, alloc->ids);
  return alloc;
}

LogicalResult DPSAnalysis::mapAlloc(ValueRange values, Allocation *alloc) {
  if (values.empty())
    return success();
  if (values.size() == 1) {
    allocViews[values[0]] = AllocView(alloc, alloc->ids);
    return success();
  }
  if (values.size() != alloc->ids.size())
    return failure();
  for (auto [value, id] : llvm::zip_equal(values, alloc->ids)) {
    allocViews[value] = AllocView(alloc, id);
  }
  return success();
}

void DPSAnalysis::print(llvm::raw_ostream &os, const SSAMap &ssaMap) const {
  auto printValue = [&](Value value) {
    os << ssaMap.lookup(value) << " = `" << ValueWithFlags(value, true) << "`";
  };

  // Sort the allocation views and provenance by their IDs to make the output
  // deterministic.
  SmallVector<std::pair<Value, int64_t>> allocOrder, provenanceOrder;
  allocOrder.reserve(allocViews.size());
  provenanceOrder.reserve(valueProvenance.size());
  ssaMap.getIds(allocViews.keys(), allocOrder);
  ssaMap.getIds(valueProvenance.keys(), provenanceOrder);
  llvm::sort(allocOrder, [](const std::pair<Value, int64_t> &lhs,
                            const std::pair<Value, int64_t> &rhs) {
    return lhs.second < rhs.second;
  });
  llvm::sort(provenanceOrder, [](const std::pair<Value, int64_t> &lhs,
                                 const std::pair<Value, int64_t> &rhs) {
    return lhs.second < rhs.second;
  });

  os << "DPS analysis {\n";
  /// Helper function to dump an allocation view.
  auto dumpAllocView = [&](Value value, AllocView allocView) {
    os << "    ";
    printValue(value);
    os << " -> " << llvm::interleaved_array(allocView.ids) << "\n";
  };
  // Print the allocation views.
  os << "  value provenance {\n";
  for (const auto &[value, id] : allocOrder) {
    AllocView allocView = getAllocView(value);
    assert(allocView.alloc && "expected allocation view to be present");
    dumpAllocView(value, allocView);
  }
  os << "  }\n";

  // Print the control-flow provenance.
  os << "  control-flow provenance {\n";

  /// Helper function to dump a provenance set.
  auto dumpProvenance = [&](Value variable, const ProvenanceSet *provenance) {
    os << "    ";
    printValue(variable);
    SmallVector<std::pair<Value, int64_t>> provenanceOrder;
    ssaMap.getIds(llvm::map_range(*provenance,
                                  [](const Provenance &provenance) {
                                    return provenance.second;
                                  }),
                  provenanceOrder);
    llvm::sort(provenanceOrder, [](const std::pair<Value, int64_t> &lhs,
                                   const std::pair<Value, int64_t> &rhs) {
      return lhs.second < rhs.second;
    });
    os << " -> {";
    llvm::interleaveComma(provenanceOrder, os,
                          [&](const std::pair<Value, int64_t> &entry) {
                            printValue(entry.first);
                          });
    os << "}\n";
  };
  for (const auto &[variable, id] : provenanceOrder) {
    const ProvenanceSet *provenance = getProvenance(variable);
    assert(provenance && "expected provenance to be present");
    dumpProvenance(variable, provenance);
  }
  os << "  }\n";
  os << "}";
}

//===----------------------------------------------------------------------===//
// DPSLiveness
//===----------------------------------------------------------------------===//

FailureOr<DPSLiveness> DPSLiveness::create(DPSAnalysis &dpsAnalysis,
                                           DataFlowSolver &solver,
                                           FunctionOpInterface funcOp) {
  DPSLiveness liveness(dpsAnalysis);
  WalkResult walkResult =
      funcOp.getFunctionBody().walk([&](InstOpInterface op) {
        ProgramPoint *afterPoint = solver.getProgramPointAfter(op);
        const LivenessState *state =
            solver.lookupState<LivenessState>(afterPoint);
        if (!state)
          return WalkResult::interrupt();
        const LivenessState::ValueSet *liveValues = state->getLiveValues();
        if (!liveValues)
          return WalkResult::interrupt();
        LiveSet &liveIds = liveness.livenessInfo[op];
        for (Value value : *liveValues) {
          DPSAnalysis::AllocView allocView = dpsAnalysis.getAllocView(value);
          assert(allocView.alloc && "expected allocation view to be present");
          liveIds.insert_range(allocView.ids);
        }
        liveness.orderedProgramPoints.push_back(op);
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return failure();
  return liveness;
}

FailureOr<bool> DPSLiveness::areAnyLive(ValueRange values,
                                        Operation *op) const {
  auto it = livenessInfo.find(op);
  if (it == livenessInfo.end())
    return failure();
  const LiveSet &liveIds = it->second;
  for (Value value : values) {
    DPSAnalysis::AllocView allocView = dpsAnalysis.getAllocView(value);
    assert(allocView.alloc && "expected allocation view to be present");
    if (llvm::any_of(allocView.ids,
                     [&](int32_t id) { return liveIds.contains(id); }))
      return true;
  }
  return false;
}

void DPSLiveness::print(llvm::raw_ostream &os, const SSAMap &ssaMap) const {
  os << "DPS liveness (after program points) {\n";
  for (Operation *op : orderedProgramPoints) {
    auto it = livenessInfo.find(op);
    assert(it != livenessInfo.end() && "expected liveness info to be present");
    const LiveSet &liveIds = it->second;
    SmallPtrSet<Value, 8> values;
    values.reserve(liveIds.size());
    for (int32_t id : liveIds) {
      const Allocation *alloc = dpsAnalysis.getAllocForId(id);
      assert(alloc && "expected allocation to be present");
      values.insert(alloc->value);
    }
    SmallVector<std::pair<Value, int64_t>> valuesWithIds;
    ssaMap.getIds(values, valuesWithIds);
    llvm::sort(valuesWithIds, [](const std::pair<Value, int64_t> &lhs,
                                 const std::pair<Value, int64_t> &rhs) {
      return lhs.second < rhs.second;
    });
    os << "  " << OpWithFlags(op, OpPrintingFlags().skipRegions()) << " -> {";
    llvm::interleaveComma(valuesWithIds, os,
                          [&](const std::pair<Value, int64_t> &entry) {
                            os << entry.second << " = `"
                               << ValueWithFlags(entry.first, true) << "`";
                          });
    os << "}\n";
  }
  os << "}";
}
