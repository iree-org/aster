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
///   - Existing iter_args (not yet supported)
/// Returns success with info.maxStage == 0 if no pipelining is needed.
static LogicalResult analyzeLoop(scf::ForOp forOp, LoopPipelineInfo &info) {
  auto cstLb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto cstUb = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto cstStep = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!cstLb || !cstUb || !cstStep)
    return forOp.emitError("aster-scf-pipeline requires constant loop bounds");

  // Collect loop bounds and step.
  info.lb = cast<IntegerAttr>(cstLb.getValue()).getInt();
  info.ub = cast<IntegerAttr>(cstUb.getValue()).getInt();
  info.step = cast<IntegerAttr>(cstStep.getValue()).getInt();
  info.numIters = (info.ub - info.lb + info.step - 1) / info.step;

  // Collect stage assignments, op program order, and maximum loop stage.
  info.maxStage = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
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
    return forOp.emitError("loop has ")
           << info.numIters << " iterations but needs at least "
           << info.maxStage + 1 << " for " << info.maxStage + 1
           << " pipeline stages";

  // Check if the loop has any results, which are not yet supported.
  if (forOp.getNumResults() > 0)
    return forOp.emitError(
        "pipelining loops with existing iter_args not yet supported");

  // TODO: stage > 0 needs IV adjustment (kernelIV - stage*step).
  if (llvm::any_of(forOp.getInductionVar().getUsers(),
                   [&forOp](Operation *user) {
                     return !forOp->isProperAncestor(user);
                   })) {
    llvm_unreachable("operation depends on the loop IV");
  }

  // Find cross-stage values: defined in stage D, used in stage U where D < U.
  for (Operation *op : info.opOrder) {
    int64_t useStage = info.stages[op];
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (v == forOp.getInductionVar())
        continue;
      auto *defOp = v.getDefiningOp();
      if (!defOp || !info.stages.count(defOp))
        continue;
      int64_t defStage = info.stages[defOp];
      if (defStage > useStage) {
        return forOp.emitError("cross-stage value ")
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
// Prologue
//===----------------------------------------------------------------------===//

/// Emit the prologue: maxStage sections that ramp up the pipeline.
///
/// Section i (0-indexed) executes stages 0..i. In section i, stage s
/// processes original iteration (i - s). This fills the pipeline so that
/// the kernel can run with all stages active.
///
/// Input: forOp (original loop), info (analysis results), builder positioned
///   before forOp.
/// Output: prologueMapping populated with all cloned results. The caller
///   extracts cross-stage values from it for kernel iter_arg initialization.
static void emitPrologue(scf::ForOp forOp, const LoopPipelineInfo &info,
                         OpBuilder &builder, IRMapping &prologueMapping) {
  Location loc = forOp.getLoc();
  for (int64_t section = 0; section < info.maxStage; ++section) {
    for (Operation *op : info.opOrder) {
      int64_t stage = info.stages.lookup(op);
      if (stage > section)
        continue;
      int64_t origIter = section - stage;
      // TODO: non-constant affine quantities.
      Value iv = arith::ConstantIndexOp::create(builder, loc,
                                                info.lb + origIter * info.step);
      prologueMapping.map(forOp.getInductionVar(), iv);
      Operation *cloned = builder.clone(*op, prologueMapping);
      prologueMapping.map(op->getResults(), cloned->getResults());
      cloned->removeAttr(kSchedStageAttr);
    }
  }
}

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

/// Emit the kernel loop: steady-state where all stages execute concurrently.
///
/// The kernel runs from lb + maxStage*step to ub. Cross-stage values are
/// carried as iter_args: stage s > 0 reads its cross-stage operands from
/// iter_args (previous iteration), while stage 0 produces new values that
/// are yielded for the next iteration.
///
/// Input: forOp, info, builder positioned after prologue, iterArgInits
///   (initial cross-stage values from prologue).
/// Output: the created kernel ForOp whose results are the final cross-stage
///   values (consumed by the epilogue).
static scf::ForOp emitKernel(scf::ForOp forOp, const LoopPipelineInfo &info,
                             OpBuilder &builder, ArrayRef<Value> iterArgInits) {
  Location loc = forOp.getLoc();
  Value kernelLb = arith::ConstantIndexOp::create(
      builder, loc, info.lb + info.maxStage * info.step);
  auto kernelLoop =
      scf::ForOp::create(builder, loc, kernelLb, forOp.getUpperBound(),
                         forOp.getStep(), iterArgInits);

  OpBuilder::InsertionGuard guard(builder);
  if (kernelLoop.getBody()->mightHaveTerminator())
    kernelLoop.getBody()->getTerminator()->erase();
  builder.setInsertionPointToStart(kernelLoop.getBody());

  // Index from cross-stage value -> iter_arg position.
  DenseMap<Value, int64_t> crossStageIdx;
  for (auto [idx, csv] : llvm::enumerate(info.crossStageVals))
    crossStageIdx[csv.value] = idx;

  // Track cloned results within the current kernel iteration.
  IRMapping kernelMapping;
  for (Operation *user : info.opOrder) {
    int64_t useStage = info.stages.lookup(user);

    IRMapping opMapping;
    opMapping.map(forOp.getInductionVar(), kernelLoop.getInductionVar());

    // Map operands to their appropriate sources:
    // - For cross-stage dependencies (def in earlier stage): use iter_args
    //   which carry values from the previous iteration.
    // - For same-stage dependencies: use results from ops cloned earlier
    //   in this iteration.
    for (Value use : user->getOperands()) {
      assert(use != forOp.getInductionVar() && "unexpected IV dependency");
      auto *defOp = use.getDefiningOp();
      if (!defOp) {
        assert(cast<BlockArgument>(use).getOwner()->getParentOp() != forOp &&
               "unexpected iter_args of the scf.for loop we are pipelining");
        continue;
      }

      // No-stage means stage 0: implicit but natural interpretation.
      if (!info.stages.count(defOp))
        continue;

      // `user` is the using op, `use` is the operand defined by `defOp` ->
      // this is the definition stage.
      int64_t defStage = info.stages.lookup(defOp);
      if (defStage < useStage) {
        auto it = crossStageIdx.find(use);
        if (it != crossStageIdx.end())
          opMapping.map(use, kernelLoop.getRegionIterArgs()[it->second]);
      } else if (Value mapped = kernelMapping.lookupOrNull(use)) {
        opMapping.map(use, mapped);
      }
    }

    // Clone the operation and map its results.
    Operation *cloned = builder.clone(*user, opMapping);
    cloned->removeAttr(kSchedStageAttr);
    kernelMapping.map(user->getResults(), cloned->getResults());
  }

  // Handle yields.
  SmallVector<Value> yieldValues;
  for (auto &csv : info.crossStageVals)
    yieldValues.push_back(kernelMapping.lookup(csv.value));
  scf::YieldOp::create(builder, loc, yieldValues);

  return kernelLoop;
}

//===----------------------------------------------------------------------===//
// Epilogue
//===----------------------------------------------------------------------===//

/// Emit the epilogue: maxStage sections that drain the pipeline.
///
/// Section j (1-indexed) executes stages j..maxStage. In section j, stage s
/// processes original iteration (numIters - s + j - 1). Cross-stage values
/// are seeded from the kernel loop results.
///
/// Input: forOp, info, builder positioned after kernel, kernelLoop (for
///   extracting final cross-stage values from its results).
/// Output: epilogue ops emitted after the kernel loop.
static void emitEpilogue(scf::ForOp forOp, const LoopPipelineInfo &info,
                         OpBuilder &builder, scf::ForOp kernelLoop) {
  Location loc = forOp.getLoc();
  IRMapping epilogueMapping;
  for (auto [idx, csv] : llvm::enumerate(info.crossStageVals))
    epilogueMapping.map(csv.value, kernelLoop.getResult(idx));

  for (int64_t epilogueStage = 1; epilogueStage <= info.maxStage;
       ++epilogueStage) {
    for (Operation *op : info.opOrder) {
      int64_t stage = info.stages.lookup(op);
      if (stage < epilogueStage)
        continue;

      int64_t origIter = info.numIters - stage + epilogueStage - 1;
      Value iv = arith::ConstantIndexOp::create(builder, loc,
                                                info.lb + origIter * info.step);
      epilogueMapping.map(forOp.getInductionVar(), iv);

      Operation *cloned = builder.clone(*op, epilogueMapping);
      cloned->removeAttr(kSchedStageAttr);
      epilogueMapping.map(op->getResults(), cloned->getResults());
    }
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
  auto walkResult = getOperation()->walk([&](scf::ForOp forOp) -> WalkResult {
    // Interrupt the walk if the loop cannot be pipelined.
    LoopPipelineInfo info;
    if (mlir::failed(analyzeLoop(forOp, info)))
      return WalkResult::interrupt();

    // Advance the walk if the loop does not need to be pipelined.
    if (info.maxStage == 0)
      return WalkResult::advance();

    LLVM_DEBUG({
      llvm::dbgs() << "Pipelining loop with " << info.maxStage + 1
                   << " stages, " << info.crossStageVals.size()
                   << " cross-stage values\n";
    });

    OpBuilder builder(forOp);

    // Step 1: Emit the prologue.
    IRMapping prologueMapping;
    emitPrologue(forOp, info, builder, prologueMapping);

    // Step 2: Map iter_args to the prologue.
    SmallVector<Value> iterArgInits;
    for (auto &csv : info.crossStageVals)
      iterArgInits.push_back(prologueMapping.lookupOrDefault(csv.value));

    // Step 3: Emit the kernel.
    auto kernelLoop = emitKernel(forOp, info, builder, iterArgInits);

    // Step 4: Emit the epilogue.
    emitEpilogue(forOp, info, builder, kernelLoop);

    // TODO: RAUW when applicable.
    forOp.erase();
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    signalPassFailure();
}

} // namespace
} // namespace mlir::aster
