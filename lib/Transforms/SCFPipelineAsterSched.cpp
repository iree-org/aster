//===- SCFPipelineAsterSched.cpp - Stage-based loop pipelining ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass pipelines scf.for loops whose operations are annotated with
// sched.stage attributes, generating prologue/kernel/epilogue sections.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

namespace mlir::aster {
#define GEN_PASS_DEF_SCFPIPELINEASTERSCHED
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

#define DEBUG_TYPE "aster-scf-pipeline"

namespace mlir::aster {
namespace {

// TODO: when stabilized, promote to a proper dialect attribute.
constexpr StringLiteral kSchedStageAttr = "sched.stage";

/// Get the pipeline stage for an operation, defaulting to 0.
static int64_t getStage(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kSchedStageAttr))
    return attr.getInt();
  return 0;
}

/// A value defined in one pipeline stage and used in a later stage.
/// These must be carried across iterations via iter_args in the kernel loop.
struct CrossStageValue {
  Value value;
  int64_t defStage;
  int64_t lastUseStage;
};

/// Analyzed loop information needed by the pipelining transform.
struct LoopPipelineInfo {
  int64_t lb, ub, step, numIters;
  int64_t maxStage;
  DenseMap<Operation *, int64_t> stages;
  SmallVector<Operation *> opOrder;
  SmallVector<CrossStageValue> crossStageVals;
};

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// Analyze a loop for pipelining feasibility and collect scheduling metadata.
///
/// Input: an scf.for loop whose body ops may carry sched.stage attributes.
/// Output: stage assignments, op program order, cross-stage values, and
/// constant loop bounds.
///
/// Returns failure with diagnostic if the loop cannot be pipelined:
///   - Non-constant bounds (static peeling only)
///   - Fewer iterations than pipeline stages
/// Returns success with info.maxStage == 0 if no pipelining is needed.
static LogicalResult analyzeLoop(scf::ForOp originalForOp,
                                 LoopPipelineInfo &info) {
  auto cstLb =
      originalForOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto cstUb =
      originalForOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto cstStep =
      originalForOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!cstLb || !cstUb || !cstStep)
    return originalForOp.emitError(
        "aster-scf-pipeline requires constant loop bounds");

  // Collect loop bounds and step.
  info.lb = cast<IntegerAttr>(cstLb.getValue()).getInt();
  info.ub = cast<IntegerAttr>(cstUb.getValue()).getInt();
  info.step = cast<IntegerAttr>(cstStep.getValue()).getInt();
  info.numIters = (info.ub - info.lb + info.step - 1) / info.step;

  // Collect stage assignments, op program order, and maximum loop stage.
  info.maxStage = 0;
  for (Operation &op : originalForOp.getBody()->without_terminator()) {
    int64_t stage = getStage(&op);
    info.stages[&op] = stage;
    info.opOrder.push_back(&op);
    info.maxStage = std::max(info.maxStage, stage);
  }

  // If no stages are assigned, no pipelining is needed.
  if (info.maxStage == 0)
    return success();

  // Check if the loop has enough iterations for the pipeline stages.
  if (info.numIters <= info.maxStage)
    return originalForOp.emitError("loop has ")
           << info.numIters << " iterations but needs at least "
           << info.maxStage + 1 << " for " << info.maxStage + 1
           << " pipeline stages";

  // Find cross-stage values: defined in stage D, used in stage U where D < U.
  // Block arguments (IV and iter_args) are not stage-defined values -- skip.
  for (Operation *op : info.opOrder) {
    int64_t useStage = info.stages[op];
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (v == originalForOp.getInductionVar())
        continue;
      // Skip iter_arg block arguments -- they are carried as existing
      // iter_args, not as cross-stage values.
      if (auto blockArg = dyn_cast<BlockArgument>(v))
        if (blockArg.getOwner() == originalForOp.getBody())
          continue;
      auto *defOp = v.getDefiningOp();
      if (!defOp || !info.stages.count(defOp))
        continue;
      int64_t defStage = info.stages[defOp];
      if (defStage > useStage) {
        return originalForOp.emitError("cross-stage value ")
               << v << " is used in stage " << useStage
               << " but defined in stage " << defStage;
      }
      if (defStage >= useStage)
        continue;
      auto it = llvm::find_if(info.crossStageVals,
                              [&](auto &c) { return c.value == v; });
      if (it != info.crossStageVals.end())
        it->lastUseStage = std::max(it->lastUseStage, useStage);
      else
        info.crossStageVals.push_back({v, defStage, useStage});
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Simulate one yield: evaluate the original yield operands through `mapping`,
/// then update iter_arg block arguments to the new values. Evaluates all
/// operands before updating to handle simultaneous swaps (yield %b, %a).
///
/// If a yield operand was not cloned (e.g., from a pipeline stage that hasn't
/// been emitted yet), the corresponding iter_arg keeps its current value.
static void simulateYield(scf::ForOp originalForOp, IRMapping &mapping) {
  if (originalForOp.getNumRegionIterArgs() == 0)
    return;
  auto yieldOp = cast<scf::YieldOp>(originalForOp.getBody()->getTerminator());
  SmallVector<Value> nextIterArgs;
  for (auto [yieldOperand, iterArg] :
       llvm::zip(yieldOp->getOperands(), originalForOp.getRegionIterArgs())) {
    if (Value mapped = mapping.lookupOrNull(yieldOperand)) {
      nextIterArgs.push_back(mapped);
    } else if (yieldOperand.getParentBlock() == originalForOp.getBody()) {
      // Defined inside loop body but not cloned -- keep current value.
      nextIterArgs.push_back(mapping.lookupOrDefault(iterArg));
    } else {
      nextIterArgs.push_back(yieldOperand);
    }
  }
  mapping.map(originalForOp.getRegionIterArgs(), nextIterArgs);
}

/// Seed an IRMapping from kernel loop results, following the iter_arg layout:
/// [cross-stage values..., existing iter_args...]
static void seedMappingFromKernelResults(scf::ForOp originalForOp,
                                         const LoopPipelineInfo &info,
                                         scf::ForOp kernelLoop,
                                         IRMapping &mapping) {
  for (auto [idx, csv] : llvm::enumerate(info.crossStageVals))
    mapping.map(csv.value, kernelLoop.getResult(idx));
  int64_t numCrossStage = info.crossStageVals.size();
  for (auto [idx, blockArg] :
       llvm::enumerate(originalForOp.getRegionIterArgs()))
    mapping.map(blockArg, kernelLoop.getResult(numCrossStage + idx));
}

/// Clone an op from the original loop body into prologue or epilogue context.
///
/// Builds a per-op IRMapping with:
///   - IV mapped to the given `iv`
///   - iter_arg block args mapped from `globalMapping` (current section state)
///   - same-origIter operands mapped from `perStageMapping` (per-stage results)
///   - everything else falls through to builder.clone's default lookup
///
/// After cloning, stores results in both `perStageMapping` and `globalMapping`,
/// and strips the sched.stage attribute.
static void cloneIntoPrologueOrEpilogue(OpBuilder &builder,
                                        Operation *originalOp,
                                        scf::ForOp originalForOp, Value iv,
                                        const LoopPipelineInfo &info,
                                        IRMapping &perStageMapping,
                                        IRMapping &globalMapping) {
  IRMapping opMapping;
  opMapping.map(originalForOp.getInductionVar(), iv);

  // Map iter_arg block arguments from current section state.
  for (auto blockArg : originalForOp.getRegionIterArgs())
    opMapping.map(blockArg, globalMapping.lookupOrDefault(blockArg));

  // Map operands from the same original iteration.
  for (Value operand : originalOp->getOperands()) {
    if (operand == originalForOp.getInductionVar())
      continue;
    auto *defOp = operand.getDefiningOp();
    if (!defOp || !info.stages.count(defOp))
      continue;
    if (Value mapped = perStageMapping.lookupOrNull(operand))
      opMapping.map(operand, mapped);
  }

  // Clone and update mappings.
  Operation *cloned = builder.clone(*originalOp, opMapping);
  cloned->removeAttr(kSchedStageAttr);
  perStageMapping.map(originalOp->getResults(), cloned->getResults());
  globalMapping.map(originalOp->getResults(), cloned->getResults());
}

//===----------------------------------------------------------------------===//
// Prologue
//===----------------------------------------------------------------------===//

/// Emit the prologue: maxStage sections that ramp up the pipeline.
///
/// Section i (0-indexed) executes stages 0..i. In section i, stage s
/// processes original iteration (i - s). This fills the pipeline so that
/// the kernel can run with all stages active.
///
/// When the original loop has existing iter_args, the prologue simulates
/// the yield after each section to advance the iter_arg values. This ensures
/// ops in later sections see the correct iter_arg values (e.g., rotated
/// offsets for LDS multi-buffering).
///
/// Input: originalForOp (original loop), info (analysis results), builder
/// positioned
///   before originalForOp.
/// Output: prologueMapping populated with all cloned results. The caller
///   extracts cross-stage values and iter_arg values from it.
static void emitPrologue(scf::ForOp originalForOp, const LoopPipelineInfo &info,
                         OpBuilder &builder, IRMapping &prologueMapping) {
  Location loc = originalForOp.getLoc();

  // Initialize iter_arg block arguments to their init values.
  prologueMapping.map(originalForOp.getRegionIterArgs(),
                      originalForOp.getInits());

  // Per-stage result mappings keeps track of the cloned results for each stage.
  // The flat prologueMapping keeps track of the current state which constantly
  // gets updated by a stage.
  SmallVector<IRMapping> perStageMappings(info.maxStage, IRMapping());

  for (int64_t section = 0; section < info.maxStage; ++section) {
    for (Operation *op : info.opOrder) {
      int64_t stage = info.stages.lookup(op);
      if (stage > section)
        continue;
      int64_t origIter = section - stage;
      assert(0 <= origIter &&
             origIter < static_cast<int64_t>(perStageMappings.size()) &&
             "origIter out of bounds");
      Value iv = arith::ConstantIndexOp::create(builder, loc,
                                                info.lb + origIter * info.step);
      cloneIntoPrologueOrEpilogue(builder, op, originalForOp, iv, info,
                                  perStageMappings[origIter], prologueMapping);
    }
    simulateYield(originalForOp, prologueMapping);
  }
}

//===----------------------------------------------------------------------===//
// Kernel helpers
//===----------------------------------------------------------------------===//

/// Build the iterArgIndex: maps original values to their position in the
/// kernel's iter_arg list. Layout: [cross-stage values..., existing
/// iter_args...]
static DenseMap<Value, int64_t>
buildIterArgIndex(scf::ForOp originalForOp, const LoopPipelineInfo &info) {
  DenseMap<Value, int64_t> iterArgIndex;
  for (auto [idx, csv] : llvm::enumerate(info.crossStageVals))
    iterArgIndex[csv.value] = idx;
  int64_t numCrossStage = info.crossStageVals.size();
  for (auto [idx, blockArg] :
       llvm::enumerate(originalForOp.getRegionIterArgs()))
    iterArgIndex[blockArg] = numCrossStage + idx;
  return iterArgIndex;
}

/// Look up a value in the iterArgIndex. Returns the corresponding kernel
/// iter_arg if found, or nullptr if the value is not an iter_arg.
static Value lookupIterArg(Value use,
                           const DenseMap<Value, int64_t> &iterArgIndex,
                           scf::ForOp kernelLoop) {
  auto it = iterArgIndex.find(use);
  if (it == iterArgIndex.end())
    return nullptr;
  return kernelLoop.getRegionIterArgs()[it->second];
}

/// Get or lazily create the stage-adjusted IV for a given pipeline stage.
/// Stage s at kernel iteration k processes original iteration
/// (k - lb)/step - s, so the effective IV is kernelIV - s * step.
static Value getOrCreateStageIV(int64_t stage, const LoopPipelineInfo &info,
                                OpBuilder &builder, scf::ForOp kernelLoop,
                                SmallVectorImpl<Value> &cache) {
  auto &iv = cache[stage];
  if (!iv) {
    Location loc = kernelLoop.getLoc();
    Value offset =
        arith::ConstantIndexOp::create(builder, loc, stage * info.step);
    iv = arith::SubIOp::create(builder, loc, kernelLoop.getInductionVar(),
                               offset);
  }
  return iv;
}

/// Build the operand mapping for one op in the kernel body.
///
/// For each operand, resolves to one of (in priority order):
///   1. Stage-adjusted IV (for the induction variable)
///   2. Kernel iter_arg (for existing iter_arg block args, or cross-stage
///      values defined in an earlier stage)
///   3. Same-iteration clone (for same-stage values cloned earlier)
///   4. Unmapped / passthrough (for values defined outside the loop)
static IRMapping mapKernelOperands(Operation *op, int64_t useStage,
                                   Value stageIV, scf::ForOp originalForOp,
                                   const LoopPipelineInfo &info,
                                   const DenseMap<Value, int64_t> &iterArgIdx,
                                   scf::ForOp kernelLoop,
                                   const IRMapping &kernelMapping) {
  IRMapping opMapping;
  opMapping.map(originalForOp.getInductionVar(), stageIV);

  for (Value use : op->getOperands()) {
    if (use == originalForOp.getInductionVar())
      continue;

    // Check iter_arg index (cross-stage values + existing iter_args).
    if (Value iterArg = lookupIterArg(use, iterArgIdx, kernelLoop)) {
      auto *defOp = use.getDefiningOp();
      // Block args (existing iter_args) always use kernel iter_args.
      if (!defOp) {
        opMapping.map(use, iterArg);
        continue;
      }
      // Cross-stage: only if defined in an earlier stage.
      if (info.stages.lookup(defOp) < useStage) {
        opMapping.map(use, iterArg);
        continue;
      }
    }

    // Same-stage: use result cloned earlier this iteration.
    auto *defOp = use.getDefiningOp();
    if (!defOp || !info.stages.count(defOp))
      continue;
    if (Value mapped = kernelMapping.lookupOrNull(use))
      opMapping.map(use, mapped);
  }

  return opMapping;
}

/// Build the yield values for the kernel loop.
///
/// Layout: [cross-stage values from this iteration...,
///          existing iter_arg yield operands resolved in kernel context...]
static SmallVector<Value>
buildKernelYieldValues(scf::ForOp originalForOp, const LoopPipelineInfo &info,
                       const DenseMap<Value, int64_t> &iterArgIdx,
                       scf::ForOp kernelLoop, const IRMapping &kernelMapping) {
  SmallVector<Value> yieldValues;

  // Cross-stage values produced in this iteration.
  for (auto &csv : info.crossStageVals)
    yieldValues.push_back(kernelMapping.lookup(csv.value));

  // Existing iter_args: resolve original yield operands in kernel context.
  auto yieldOp = cast<scf::YieldOp>(originalForOp.getBody()->getTerminator());
  for (Value yieldOperand : yieldOp->getOperands()) {
    if (Value iterArg = lookupIterArg(yieldOperand, iterArgIdx, kernelLoop))
      yieldValues.push_back(iterArg);
    else if (Value mapped = kernelMapping.lookupOrNull(yieldOperand))
      yieldValues.push_back(mapped);
    else
      yieldValues.push_back(yieldOperand); // outside loop
  }

  return yieldValues;
}

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

/// Emit the kernel loop: steady-state where all stages execute concurrently.
///
/// The kernel runs from lb + maxStage*step to ub. All stages execute each
/// iteration, with cross-stage values carried as iter_args from the previous
/// iteration. Existing iter_args are appended after cross-stage iter_args.
///
/// Iter_arg layout: [cross-stage values..., existing iter_args...]
static scf::ForOp emitKernel(scf::ForOp originalForOp,
                             const LoopPipelineInfo &info, OpBuilder &builder,
                             SmallVectorImpl<Value> &iterArgInits) {
  Location loc = originalForOp.getLoc();
  Value kernelLb = arith::ConstantIndexOp::create(
      builder, loc, info.lb + info.maxStage * info.step);
  auto kernelLoop =
      scf::ForOp::create(builder, loc, kernelLb, originalForOp.getUpperBound(),
                         originalForOp.getStep(), iterArgInits);

  OpBuilder::InsertionGuard guard(builder);
  if (kernelLoop.getBody()->mightHaveTerminator())
    kernelLoop.getBody()->getTerminator()->erase();
  builder.setInsertionPointToStart(kernelLoop.getBody());

  auto iterArgIdx = buildIterArgIndex(originalForOp, info);
  SmallVector<Value> stageIVCache(info.maxStage + 1);
  stageIVCache[0] = kernelLoop.getInductionVar();

  // Clone each op with stage-adjusted IV and resolved operands.
  IRMapping kernelMapping;
  for (Operation *op : info.opOrder) {
    int64_t stage = info.stages.lookup(op);
    Value iv =
        getOrCreateStageIV(stage, info, builder, kernelLoop, stageIVCache);
    auto opMapping = mapKernelOperands(op, stage, iv, originalForOp, info,
                                       iterArgIdx, kernelLoop, kernelMapping);
    Operation *cloned = builder.clone(*op, opMapping);
    cloned->removeAttr(kSchedStageAttr);
    kernelMapping.map(op->getResults(), cloned->getResults());
  }

  auto yieldValues = buildKernelYieldValues(originalForOp, info, iterArgIdx,
                                            kernelLoop, kernelMapping);
  scf::YieldOp::create(builder, loc, yieldValues);

  return kernelLoop;
}

//===----------------------------------------------------------------------===//
// Epilogue
//===----------------------------------------------------------------------===//

/// Emit the epilogue: maxStage sections that drain the pipeline.
/// Section j (1-indexed) executes stages j..maxStage, where stage s processes
/// original iteration (numIters - s + j - 1). Cross-stage values are seeded
/// from the kernel results; yields are simulated between sections.
///
/// epilogueMapping is populated with the final iter_arg values after all
/// epilogue sections, for use in replacing the original loop results.
static void emitEpilogue(scf::ForOp originalForOp, const LoopPipelineInfo &info,
                         OpBuilder &builder, scf::ForOp kernelLoop,
                         IRMapping &epilogueMapping) {
  Location loc = originalForOp.getLoc();
  seedMappingFromKernelResults(originalForOp, info, kernelLoop,
                               epilogueMapping);

  // Per-original-iteration result mappings. Cross-stage dependencies require
  // finding the value produced at the same logical iteration.
  // Seed: at kernel exit, cross-stage iter_args hold values produced by
  // defStage at origIter = (numIters - 1) - csv.defStage.
  DenseMap<int64_t, IRMapping> perStageMappings;
  for (auto [idx, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t defOrigIter = (info.numIters - 1) - csv.defStage;
    perStageMappings[defOrigIter].map(csv.value, kernelLoop.getResult(idx));
  }

  for (int64_t epilogueStage = 1; epilogueStage <= info.maxStage;
       ++epilogueStage) {
    for (Operation *op : info.opOrder) {
      int64_t stage = info.stages.lookup(op);
      if (stage < epilogueStage)
        continue;
      int64_t origIter = info.numIters - stage + epilogueStage - 1;
      Value iv = arith::ConstantIndexOp::create(builder, loc,
                                                info.lb + origIter * info.step);
      cloneIntoPrologueOrEpilogue(builder, op, originalForOp, iv, info,
                                  perStageMappings[origIter], epilogueMapping);
    }
    simulateYield(originalForOp, epilogueMapping);
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SCFPipelineAsterSchedPass
    : public impl::SCFPipelineAsterSchedBase<SCFPipelineAsterSchedPass> {
  using Base::Base;
  void runOnOperation() override;
};

void SCFPipelineAsterSchedPass::runOnOperation() {
  auto walkResult =
      getOperation()->walk([&](scf::ForOp originalForOp) -> WalkResult {
        // Interrupt the walk if the loop cannot be pipelined.
        LoopPipelineInfo info;
        if (mlir::failed(analyzeLoop(originalForOp, info)))
          return WalkResult::interrupt();

        // Advance the walk if the loop does not need to be pipelined.
        if (info.maxStage == 0)
          return WalkResult::advance();

        LLVM_DEBUG({
          llvm::dbgs() << "Pipelining loop with " << info.maxStage + 1
                       << " stages, " << info.crossStageVals.size()
                       << " cross-stage values\n";
        });

        OpBuilder builder(originalForOp);

        // Step 1: Emit the prologue.
        IRMapping prologueMapping;
        emitPrologue(originalForOp, info, builder, prologueMapping);

        // Step 2: Collect kernel iter_arg initial values.
        // Layout: [cross-stage values..., existing iter_args...]
        SmallVector<Value> iterArgInits;
        for (auto &csv : info.crossStageVals)
          iterArgInits.push_back(prologueMapping.lookupOrDefault(csv.value));
        // Existing iter_args: use the values from after prologue yield
        // simulation.
        for (auto blockArg : originalForOp.getRegionIterArgs())
          iterArgInits.push_back(prologueMapping.lookupOrDefault(blockArg));

        // Step 3: Emit the kernel.
        auto kernelLoop =
            emitKernel(originalForOp, info, builder, iterArgInits);

        IRMapping epilogueMapping;
        emitEpilogue(originalForOp, info, builder, kernelLoop, epilogueMapping);

        // Replace original loop results with final epilogue iter_arg values.
        for (auto [oldResult, blockArg] : llvm::zip(
                 originalForOp.getResults(), originalForOp.getRegionIterArgs()))
          oldResult.replaceAllUsesWith(
              epilogueMapping.lookupOrDefault(blockArg));

        originalForOp.erase();
        return WalkResult::advance();
      });

  if (walkResult.wasInterrupted())
    signalPassFailure();
}

} // namespace
} // namespace mlir::aster
