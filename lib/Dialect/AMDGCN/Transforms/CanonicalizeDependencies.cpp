//===- CanonicalizeDependencies.cpp - Canonicalize dependency tokens -----===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DependencyAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNCANONICALIZEDEPENDENCIES
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct AMDGCNCanonicalizeDependencies
    : public mlir::aster::amdgcn::impl::AMDGCNCanonicalizeDependenciesBase<
          AMDGCNCanonicalizeDependencies> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static bool areCompatibleMemInstKind(amdgcn::MemoryInstructionKind a,
                                     amdgcn::MemoryInstructionKind b) {
  if (a == b)
    return true;
  if (a == amdgcn::MemoryInstructionKind::Constant)
    std::swap(a, b);
  return (a == amdgcn::MemoryInstructionKind::Shared &&
          b == amdgcn::MemoryInstructionKind::Constant);
}

static bool areCompatibleDependencyTokens(Value a, Value b) {
  auto getKind = [](Value v) -> std::optional<amdgcn::MemoryInstructionKind> {
    if (auto wTy = dyn_cast<amdgcn::WriteTokenType>(v.getType()))
      return wTy.getKind();
    if (auto rTy = dyn_cast<amdgcn::ReadTokenType>(v.getType()))
      return rTy.getKind();
    return std::nullopt;
  };
  std::optional<amdgcn::MemoryInstructionKind> aKind = getKind(a);
  std::optional<amdgcn::MemoryInstructionKind> bKind = getKind(b);
  return aKind && bKind && areCompatibleMemInstKind(*aKind, *bKind);
}

void AMDGCNCanonicalizeDependencies::runOnOperation() {
  Operation *op = getOperation();

  // Configure dataflow solver with dependency analysis.
  DataFlowSolver solver;
  solver.load<DependencyAnalysis>();
  dataflow::loadBaselineAnalyses(solver);
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  // Retrieve dominance analysis.
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();

  // Collect all wait operations.
  // FIXME: The update requires fixpoint.
  WalkResult result = op->walk([&](amdgcn::WaitOp waitOp) -> WalkResult {
    ValueRange dependencies = waitOp.getDependencies();
    MutableOperandRange depsMutable = waitOp.getDependenciesMutable();
    MutableOperandRange passthrough = waitOp.getPassthroughMutable();
    int64_t size = passthrough.size();
    int64_t i = 0;
    // Remove from passthrough any tokens that are in dependencies.
    while (i < size && !passthrough.empty()) {
      if (llvm::count(dependencies, passthrough[i].get()) == 0) {
        ++i;
        continue;
      }
      passthrough.erase(i);
      --size;
    }
    auto *state = solver.lookupState<DependencyState>(
        solver.getProgramPointBefore(waitOp));
    assert(state && "DependencyState should be available");
    if (state->isTop()) {
      waitOp.emitError("Found top state");
      return WalkResult::interrupt();
    }
    const llvm::SmallDenseSet<Value> &pendingTokens =
        *(state->getPendingTokens());

    // Remove from dependencies any tokens that are not pending as they were
    // already resolved.
    i = 0;
    size = depsMutable.size();
    while (i < size && !depsMutable.empty()) {
      if (pendingTokens.contains(depsMutable[i].get())) {
        ++i;
        continue;
      }
      depsMutable.erase(i);
      --size;
    }

    // If dependencies are empty, remove the wait operation.
    if (depsMutable.empty()) {
      waitOp.erase();
      return WalkResult::advance();
    }

    // Add any pending tokens that dominate existing dependencies.
    auto *afterState = solver.lookupState<DependencyState>(
        solver.getProgramPointBefore(waitOp));
    assert(afterState && "DependencyState should be available");
    if (afterState->isTop()) {
      waitOp.emitError("Found top state");
      return WalkResult::interrupt();
    }
    const llvm::SmallDenseSet<Value> &afterTokens =
        *(afterState->getPendingTokens());
    llvm::SetVector<Value> extraDeps;
    dependencies = waitOp.getDependencies();
    for (Value v : afterTokens) {
      for (Value dep : dependencies) {
        if (v == dep || !areCompatibleDependencyTokens(v, dep))
          continue;
        // FIXME: The values might come from a block.
        if (domInfo.properlyDominates(v.getDefiningOp(), dep.getDefiningOp()))
          extraDeps.insert(v);
      }
    }
    depsMutable.append(extraDeps.getArrayRef());

    // Add to passthrough any tokens that were not added to dependencies.
    for (Value v : afterTokens) {
      if (extraDeps.contains(v))
        continue;
      passthrough.append(v);
    }

    waitOp.getProperties().operandSegmentSizes = {
        static_cast<int>(depsMutable.size()),
        static_cast<int>(passthrough.size()),
    };
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
