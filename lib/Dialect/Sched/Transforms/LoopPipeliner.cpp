//===- LoopPipeliner.cpp - Pipeline a loop around a target stage --*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass pipelines a loop that contains sched.unit ops by restructuring it
// into prologue, main body, and epilogue sections around a target stage.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/Sched/IR/SchedOps.h"
#include "aster/Dialect/Sched/Transforms/Passes.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster::sched {
#define GEN_PASS_DEF_LOOPPIPELINER
#include "aster/Dialect/Sched/Transforms/Passes.h.inc"
} // namespace mlir::aster::sched

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::sched;
using namespace mlir::aster::aster_utils;

namespace {

/// Returns the stage number of a sched.unit op.
static std::optional<int64_t> getUnitStage(UnitOp unit) {
  APInt stageVal = unit.getStage();
  return stageVal.getSExtValue();
}

/// Collect all sched.unit ops directly inside the first region of the loop.
static SmallVector<UnitOp> collectUnits(LoopLikeOpInterface loop) {
  SmallVector<UnitOp> units;
  for (Region &region : loop->getRegions())
    for (Block &block : region)
      for (Operation &op : block)
        if (UnitOp unit = dyn_cast<UnitOp>(&op))
          units.push_back(unit);
  return units;
}

/// Build a stage -> UnitOp map and identify prologue/epilogue stage sets.
static bool classifyStages(ArrayRef<UnitOp> units, int64_t tgtStage,
                           DenseMap<int64_t, UnitOp> &stageToUnit,
                           SmallVectorImpl<int64_t> &prologueStages,
                           SmallVectorImpl<int64_t> &epilogueStages) {
  for (UnitOp unit : units) {
    std::optional<int64_t> stage = getUnitStage(unit);
    if (!stage)
      return false;
    stageToUnit[*stage] = unit;
  }

  SmallVector<int64_t> allStages;
  for (auto &[stage, _] : stageToUnit)
    allStages.push_back(stage);
  llvm::sort(allStages);

  // Stages ≤ tgtStage are prologue stages; stages > tgtStage are epilogue.
  for (int64_t s : allStages) {
    if (s <= tgtStage)
      prologueStages.push_back(s);
    else
      epilogueStages.push_back(s);
  }
  return true;
}

/// Clone a unit op's body at the given insertion point, mapping block args
/// from the unit's operands via `inputMapping`.
static SmallVector<Value> emitUnitBody(UnitOp unit, OpBuilder &builder,
                                       IRMapping &mapping) {
  Block &unitBlock = unit.getBody().front();
  // Map block arguments to the (already mapped) unit inputs.
  for (auto [arg, input] :
       llvm::zip(unitBlock.getArguments(), unit.getInputs()))
    mapping.map(arg, mapping.lookupOrDefault(input));

  for (Operation &op : unitBlock.without_terminator())
    builder.clone(op, mapping);

  YieldOp yield = cast<YieldOp>(unitBlock.getTerminator());
  SmallVector<Value> results;
  for (Value v : yield.getValues())
    results.push_back(mapping.lookupOrDefault(v));
  return results;
}

struct LoopPipelinerPass
    : public mlir::aster::sched::impl::LoopPipelinerBase<LoopPipelinerPass> {
  using LoopPipelinerBase::LoopPipelinerBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    SmallVector<LoopLikeOpInterface> loops;
    root->walk([&](LoopLikeOpInterface loop) {
      if (!collectUnits(loop).empty())
        loops.push_back(loop);
    });

    // Set up the SchedAnalysis infrastructure (needed by SSASchedulerAttr).
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
    AnalysisManager analysisManager = getAnalysisManager();
    SchedAnalysis schedAnalysis(root, solver, domInfo, analysisManager);

    for (LoopLikeOpInterface loop : loops)
      if (failed(pipelineLoop(loop, schedAnalysis)))
        return signalPassFailure();
  }

  LogicalResult pipelineLoop(LoopLikeOpInterface loop,
                             SchedAnalysis &schedAnalysis) {
    SmallVector<UnitOp> units = collectUnits(loop);
    if (units.empty())
      return success();

    // Build the SSA dependency graph among ops in the first loop block using
    // SSASchedulerAttr (from aster_utils).
    Block *loopBlock = &loop->getRegion(0).front();
    SSASchedulerAttr scheduler = SSASchedulerAttr::get(loop->getContext());
    FailureOr<SchedGraph> graphOrErr =
        scheduler.createGraph(loopBlock, schedAnalysis);
    if (failed(graphOrErr))
      return loop->emitOpError("failed to build SSA dependency graph");

    // Classify stages.
    DenseMap<int64_t, UnitOp> stageToUnit;
    SmallVector<int64_t> prologueStages, epilogueStages;
    if (!classifyStages(units, tgtStage, stageToUnit, prologueStages,
                        epilogueStages))
      return loop->emitOpError("sched.unit ops are missing stage attributes");

    if (prologueStages.empty())
      return loop->emitOpError("no prologue stages found for target stage ")
             << tgtStage;

    // Emit the prologue: clone prologue stages before the loop.
    // The prologue represents iteration 0's prologue stages.
    OpBuilder builder(loop.getOperation());
    IRMapping prologueMapping;

    // Map values that come from outside the loop (loop's operands are already
    // in scope). We don't remap them: they are available as-is.
    for (int64_t stage : prologueStages) {
      UnitOp unit = stageToUnit[stage];
      SmallVector<Value> results = emitUnitBody(unit, builder, prologueMapping);
      // Map each unit result (by the original unit op) so subsequent stages
      // can use them.
      for (auto [orig, res] : llvm::zip(unit.getResults(), results))
        prologueMapping.map(orig, res);
    }

    // The epilogue: clone epilogue stages after the loop.
    builder.setInsertionPointAfter(loop.getOperation());
    IRMapping epilogueMapping;
    // Epilogue values come from the prologue mapping (last iteration's values).
    for (auto &[val, mapped] : prologueMapping.getValueMap())
      epilogueMapping.map(val, mapped);

    for (int64_t stage : epilogueStages) {
      UnitOp unit = stageToUnit[stage];
      SmallVector<Value> results = emitUnitBody(unit, builder, epilogueMapping);
      for (auto [orig, res] : llvm::zip(unit.getResults(), results))
        epilogueMapping.map(orig, res);
    }

    // Restructure the loop body:
    //   1. Move epilogue stages to the beginning of the body.
    //   2. Keep prologue stages + tgtStage for the "next" iteration.
    // This requires loop iter_arg manipulation which is loop-specific.
    // For now, emit a note and leave the loop body as-is. A full
    // implementation would use LoopLikeOpInterface::replaceWithAdditionalYields
    // or similar to thread inter-iteration values.
    //
    // TODO: implement full loop body restructuring and iter_arg threading.

    return success();
  }
};

} // namespace
