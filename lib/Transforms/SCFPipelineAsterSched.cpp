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
// The algorithm maintains different mappings to keep track of values:
//   (a) a "latest value" mapping keeps track of the current state of every
//   original value in the pipeline. Each clone updates the latest value mapping
//   with the new value.
//   (b) a "per stage" mapping keeps track of the results of each stage. This is
//   useful when cross-stage values have gaps > 1 and require "older" versions
//   of a value.
//   (c) an "epilogue" mapping to track the results of the kernel loop and how
//   it is used in the draining epilogue after all stages have completed.
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Transforms/Passes.h"
#include "aster/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <numeric>

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
///
/// When the stage gap (lastUseStage - defStage) exceeds 1, a shift of that
/// depth is needed: each cross-stage value occupies `distance()` consecutive
/// iter_args, forming a pipeline that shifts values forward.
struct CrossStageValue {
  Value value;
  int64_t defStage;
  int64_t lastUseStage;

  /// Number of iterations the value must survive as an iter_arg.
  /// For consecutive stages this is 1; for gaps it equals the gap size.
  int64_t distance() const { return lastUseStage - defStage; }
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
  auto cstStep =
      originalForOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!cstLb || !cstStep)
    return originalForOp.emitError(
        "aster-scf-pipeline requires constant lower bound and step");

  // Collect loop bounds and step. Upper bound may be dynamic.
  info.lb = cast<IntegerAttr>(cstLb.getValue()).getInt();
  info.step = cast<IntegerAttr>(cstStep.getValue()).getInt();
  auto cstUb =
      originalForOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  if (cstUb) {
    info.ub = cast<IntegerAttr>(cstUb.getValue()).getInt();
    info.numIters = (info.ub - info.lb + info.step - 1) / info.step;
  } else {
    info.ub = -1;
    info.numIters = -1;
  }

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
  // Skip when upper bound is dynamic (runtime responsibility).
  if (info.numIters > 0 && info.numIters <= info.maxStage)
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

/// Compute the total number of iter_args needed for cross-stage values.
/// Each cross-stage value with distance D occupies D consecutive iter_arg
/// slots.
// TODO: consider just grouping everything with a struct.
static int64_t totalCrossStageIterArgs(const LoopPipelineInfo &info) {
  int64_t total = 0;
  for (auto &csv : info.crossStageVals)
    total += csv.distance();
  return total;
}

/// Compute the base iter_arg index for the i-th cross-stage value.
/// Each cross-stage value occupies csv.distance() consecutive slots.
// TODO: consider just grouping everything with a struct.
static int64_t crossStageBaseIndex(const LoopPipelineInfo &info, int64_t i) {
  int64_t base = 0;
  for (int64_t j = 0; j < i; ++j)
    base += info.crossStageVals[j].distance();
  return base;
}

/// Seed an IRMapping from kernel loop results, following the iter_arg layout:
/// [cross-stage shift values..., existing iter_args...]
///
/// For each cross-stage value with distance D, the last slot (base + D - 1)
/// holds the oldest value, i.e. the one the use-stage consumes.
/// This last slot is mapped to csv.value.
// TODO: consider just grouping everything with a struct.
static void seedMappingFromKernelResults(scf::ForOp originalForOp,
                                         const LoopPipelineInfo &info,
                                         scf::ForOp kernelLoop,
                                         IRMapping &mapping) {
  for (auto [i, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t base = crossStageBaseIndex(info, i);
    // The oldest slot (base + D - 1) is what the use-stage reads.
    mapping.map(csv.value, kernelLoop.getResult(base + csv.distance() - 1));
  }
  int64_t numCSIterArgs = totalCrossStageIterArgs(info);
  for (auto [idx, blockArg] :
       llvm::enumerate(originalForOp.getRegionIterArgs()))
    mapping.map(blockArg, kernelLoop.getResult(numCSIterArgs + idx));
}

/// Clone an op from the original loop body into prologue or epilogue context.
///
/// Builds a per-op IRMapping with:
///   - IV mapped to the given `iv`
///   - iter_arg block args mapped from `latestValueMapping` (current section
///   state)
///   - same-origIter operands mapped from `perStageMapping` (per-stage results)
///   - everything else falls through to builder.clone's default lookup
///
/// After cloning, stores results in both `perStageMapping` and
/// `latestValueMapping`, and strips the sched.stage attribute.
static void cloneIntoPrologueOrEpilogue(OpBuilder &builder,
                                        Operation *originalOp,
                                        scf::ForOp originalForOp, Value iv,
                                        const LoopPipelineInfo &info,
                                        IRMapping &perStageMapping,
                                        IRMapping &latestValueMapping) {
  IRMapping opMapping;
  opMapping.map(originalForOp.getInductionVar(), iv);

  // Map iter_arg block arguments from current section state.
  for (auto blockArg : originalForOp.getRegionIterArgs())
    opMapping.map(blockArg, latestValueMapping.lookupOrDefault(blockArg));

  // Map operands: prefer same-origIter (perStageMapping), fall back to
  // cross-stage (latestValueMapping) for values from earlier stages that
  // finished in an earlier section.
  for (Value operand : originalOp->getOperands()) {
    if (operand == originalForOp.getInductionVar())
      continue;
    auto *defOp = operand.getDefiningOp();
    if (!defOp || !info.stages.count(defOp))
      continue;
    if (Value mapped = perStageMapping.lookupOrNull(operand))
      opMapping.map(operand, mapped);
    else if (Value mapped = latestValueMapping.lookupOrNull(operand))
      opMapping.map(operand, mapped);
    else
      llvm_unreachable("operand not found in perStage or latestValue mappings");
  }

  // Clone and update mappings.
  Operation *cloned = builder.clone(*originalOp, opMapping);
  cloned->removeAttr(kSchedStageAttr);
  perStageMapping.map(originalOp->getResults(), cloned->getResults());
  latestValueMapping.map(originalOp->getResults(), cloned->getResults());
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
/// Output:
///   - latestValueMapping: global mapping with latest cloned results
///   - perStageMappings: per-stage results, needed for kernel init
///     when cross-stage values have gaps > 1
static void emitPrologue(scf::ForOp originalForOp, const LoopPipelineInfo &info,
                         OpBuilder &builder, IRMapping &latestValueMapping,
                         SmallVector<IRMapping> &perStageMappings) {
  Location loc = originalForOp.getLoc();

  // Initialize iter_arg block arguments to their init values.
  latestValueMapping.map(originalForOp.getRegionIterArgs(),
                         originalForOp.getInits());

  // Per-stage result mappings keeps track of the cloned results for each stage.
  // The flat latestValueMapping keeps track of the current state which
  // constantly gets updated by a stage.
  perStageMappings.assign(info.maxStage, IRMapping());

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
                                  perStageMappings[origIter],
                                  latestValueMapping);
    }
    simulateYield(originalForOp, latestValueMapping);
  }
}

//===----------------------------------------------------------------------===//
// Kernel helpers
//===----------------------------------------------------------------------===//

/// Build the iterArgIndex: maps original values to the kernel iter_arg slot
/// that the use-stage reads from (the oldest slot in the shift register).
///
/// Layout: [csv0 shift reg (D0 slots)..., csv1 shift reg (D1 slots)...,
///          existing iter_args...]
///
/// For each CSV with distance D, the use-stage reads from slot (base + D - 1).
// TODO: consider just grouping everything with a struct and simplify the logic.
static DenseMap<Value, int64_t>
buildIterArgIndex(scf::ForOp originalForOp, const LoopPipelineInfo &info) {
  DenseMap<Value, int64_t> iterArgIndex;
  for (auto [i, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t base = crossStageBaseIndex(info, i);
    iterArgIndex[csv.value] = base + csv.distance() - 1;
  }
  int64_t numCSIterArgs = totalCrossStageIterArgs(info);
  for (auto [idx, blockArg] :
       llvm::enumerate(originalForOp.getRegionIterArgs()))
    iterArgIndex[blockArg] = numCSIterArgs + idx;
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
    // iv = kernelIV - stage * step
    auto map =
        AffineMap::get(1, 0, builder.getAffineDimExpr(0) - stage * info.step);
    iv = affine::AffineApplyOp::create(builder, loc, map,
                                       kernelLoop.getInductionVar());
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
static IRMapping mapKernelOperands(
    Operation *op, int64_t useStage, Value stageIV, scf::ForOp originalForOp,
    const LoopPipelineInfo &info, const DenseMap<Value, int64_t> &iterArgIdx,
    const DenseMap<Value, std::pair<int64_t, int64_t>> &crossStageSlotInfo,
    scf::ForOp kernelLoop, const IRMapping &kernelMapping) {
  IRMapping opMapping;
  opMapping.map(originalForOp.getInductionVar(), stageIV);

  for (Value use : op->getOperands()) {
    if (use == originalForOp.getInductionVar())
      continue;

    // Existing iter_args (block arguments) always use kernel iter_args.
    if (Value iterArg = lookupIterArg(use, iterArgIdx, kernelLoop)) {
      if (!use.getDefiningOp()) {
        opMapping.map(use, iterArg);
        continue;
      }
    }

    // Cross-stage values: compute the correct shift-register slot based on
    // the use stage. A value with shift register at baseIndex, defined at
    // defStage, read at useStage maps to:
    //   slot = baseIndex + (useStage - defStage) - 1
    auto csIt = crossStageSlotInfo.find(use);
    if (csIt != crossStageSlotInfo.end()) {
      auto [base, defStage] = csIt->second;
      if (defStage < useStage) {
        int64_t slot = base + (useStage - defStage) - 1;
        opMapping.map(use, kernelLoop.getRegionIterArgs()[slot]);
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
/// Layout: [csv0 (D0 slots for shifts)..., csv1 (D1 slots for shifts)...,
///          existing iter_arg yield operands resolved in kernel context...]
///
/// For each cross-stage value with distance D, the kernel yields:
///   slot 0: freshly produced value (from this iteration's defStage)
///   slot 1: old iter_arg[0]  (shifts forward)
///   ...
///   slot D-2: old iter_arg[D-3]
/// The old slot D-1 (consumed this iteration) is dropped.
// TODO: consider just grouping everything with a struct and simplify the logic.
static SmallVector<Value>
buildKernelYieldValues(scf::ForOp originalForOp, const LoopPipelineInfo &info,
                       const DenseMap<Value, int64_t> &iterArgIdx,
                       scf::ForOp kernelLoop, const IRMapping &kernelMapping) {
  SmallVector<Value> yieldValues;

  // Cross-stage shift register yields.
  for (auto [i, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t base = crossStageBaseIndex(info, i);
    int64_t D = csv.distance();
    // Slot 0: freshly produced value.
    yieldValues.push_back(kernelMapping.lookup(csv.value));
    // Slots 1..D-1: shift from previous slots.
    for (int64_t d = 1; d < D; ++d)
      yieldValues.push_back(kernelLoop.getRegionIterArgs()[base + d - 1]);
  }

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

  // Build per-use-stage slot info for cross-stage values. Maps each value
  // to {baseIndex, defStage} so mapKernelOperands can compute the correct
  // shift-register slot for each use stage.
  DenseMap<Value, std::pair<int64_t, int64_t>> crossStageSlotInfo;
  for (auto [i, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t base = crossStageBaseIndex(info, i);
    crossStageSlotInfo[csv.value] = {base, csv.defStage};
  }

  SmallVector<Value> stageIVCache(info.maxStage + 1);
  stageIVCache[0] = kernelLoop.getInductionVar();

  // Clone each op with stage-adjusted IV and resolved operands.
  IRMapping kernelMapping;
  for (Operation *op : info.opOrder) {
    int64_t stage = info.stages.lookup(op);
    Value iv =
        getOrCreateStageIV(stage, info, builder, kernelLoop, stageIVCache);
    auto opMapping =
        mapKernelOperands(op, stage, iv, originalForOp, info, iterArgIdx,
                          crossStageSlotInfo, kernelLoop, kernelMapping);
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
  //
  // Seed from kernel exit: for each CSV with distance D, the shift register
  // holds D values. Slot d (0-indexed) was produced at
  //   origIter = (numIters - 1) - csv.defStage - d
  // (slot 0 = freshest, slot D-1 = oldest = what use-stage last consumed).
  DenseMap<int64_t, IRMapping> perStageMappings;
  for (auto [i, csv] : llvm::enumerate(info.crossStageVals)) {
    int64_t base = crossStageBaseIndex(info, i);
    int64_t D = csv.distance();
    for (int64_t d = 0; d < D; ++d) {
      int64_t defOrigIter = (info.numIters - 1) - csv.defStage - d;
      perStageMappings[defOrigIter].map(csv.value,
                                        kernelLoop.getResult(base + d));
    }
  }

  Value ubValue = originalForOp.getUpperBound();
  for (int64_t epilogueStage = 1; epilogueStage <= info.maxStage;
       ++epilogueStage) {
    for (Operation *op : info.opOrder) {
      int64_t stage = info.stages.lookup(op);
      if (stage < epilogueStage)
        continue;
      int64_t origIter = info.numIters > 0
                             ? info.numIters - stage + epilogueStage - 1
                             : -(stage - epilogueStage + 1);
      // IV = ub - (stage - epilogueStage + 1) * step
      // Works for both constant and dynamic upper bounds.
      int64_t offset = (stage - epilogueStage + 1) * info.step;
      auto map = AffineMap::get(1, 0, builder.getAffineDimExpr(0) - offset);
      Value iv = affine::AffineApplyOp::create(builder, loc, map, ubValue);
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

/// Compute the GCD of all distinct nonzero stage values in the loop.
static int64_t computeStageGCD(const LoopPipelineInfo &info) {
  int64_t g = 0;
  for (auto [_, stage] : info.stages) {
    if (stage > 0)
      g = std::gcd(g, stage);
  }
  return g > 0 ? g : 1;
}

void SCFPipelineAsterSchedPass::runOnOperation() {
  // Prepare LDS buffers for multi-buffering before pipelining. This hoists
  // alloc_lds ops out of loops and adds rotating offset iter_args, so the
  // pipeliner sees only generic iter_args with no LDS-specific logic.
  if (failed(prepareLDSMultibuffers(getOperation())))
    return signalPassFailure();

  // Collect kernel loops to unroll after the walk (unrolling mutates IR).
  SmallVector<std::pair<scf::ForOp, int64_t>> loopsToUnroll;

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
        IRMapping latestValueMapping;
        SmallVector<IRMapping> prologuePerStageMappings;
        emitPrologue(originalForOp, info, builder, latestValueMapping,
                     prologuePerStageMappings);

        // Step 2: Collect kernel iter_arg initial values.
        // Layout: [csv0 (D0 slots for shifts)..., existing iter_args...]
        //
        // For each cross-stage value with distance D, the kernel's first
        // iteration (ki = maxStage) has the use-stage processing:
        //   `origIter = maxStage - lastUseStage`
        // It needs the value from that origIter, which was produced in the
        // prologue.
        // The shifted values are initialized with D consecutive prologue
        // values: the freshest at slot 0, the oldest at slot D-1.
        // TODO: consider just grouping everything with a struct and simplify
        // the logic.
        SmallVector<Value> iterArgInits;
        for (auto &csv : info.crossStageVals) {
          int64_t D = csv.distance();
          // At kernel start, the shifted values should contain the D most
          // recent values from the prologue.
          // The oldest (slot D-1) was produced at
          //   `origIter = maxStage - lastUseStage`,
          // and the freshest (slot 0) was produced at
          //   `origIter = maxStage - defStage - 1`.
          for (int64_t d = 0; d < D; ++d) {
            // Slot d holds value from origIter = (maxStage - defStage - 1) - d
            int64_t origIter = (info.maxStage - csv.defStage - 1) - d;
            assert(origIter >= 0 &&
                   origIter <
                       static_cast<int64_t>(prologuePerStageMappings.size()) &&
                   "shift register origIter out of prologue range");
            iterArgInits.push_back(
                prologuePerStageMappings[origIter].lookupOrDefault(csv.value));
          }
        }
        // Existing iter_args: use the values from after prologue yield
        // simulation.
        for (auto blockArg : originalForOp.getRegionIterArgs())
          iterArgInits.push_back(latestValueMapping.lookupOrDefault(blockArg));

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

        if (gcdUnroll) {
          int64_t factor = computeStageGCD(info);
          if (factor > 1)
            loopsToUnroll.push_back({kernelLoop, factor});
        }

        return WalkResult::advance();
      });

  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // TODO: consider adding a Duff device if it helps regalloc + nowait.
  for (auto [loop, factor] : loopsToUnroll) {
    if (failed(mlir::loopUnrollByFactor(loop, factor)))
      return signalPassFailure();
  }
}

} // namespace
} // namespace mlir::aster
