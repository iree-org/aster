//===- GroupStagesToUnits.cpp - Group staged ops into sched.unit --*- C++
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
// This pass groups operations inside loop bodies by their sched.stage
// discardable attribute into sched.unit ops. SSA-dependent unannotated ops are
// assigned to the earliest dominating stage that consumes their results.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Sched/IR/SchedOps.h"
#include "aster/Dialect/Sched/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster::sched {
#define GEN_PASS_DEF_GROUPSTAGESTOUNITS
#include "aster/Dialect/Sched/Transforms/Passes.h.inc"
} // namespace mlir::aster::sched

using namespace mlir;
using namespace mlir::aster::sched;

namespace {

static constexpr StringLiteral kSchedStageAttr = "sched.stage";

/// Returns the stage number for an op if it has a sched.stage discardable attr.
static std::optional<int64_t> getStage(Operation *op) {
  Attribute attr = op->getDiscardableAttr(kSchedStageAttr);
  if (!attr)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

/// Process a single loop body block, grouping ops into sched.unit ops by stage.
static LogicalResult processBlock(Block &block, OpBuilder &builder) {
  // Collect stage-annotated ops and their stages (in program order).
  llvm::MapVector<int64_t, SmallVector<Operation *>> stageOps;
  SmallVector<Operation *> allOps;

  for (Operation &op : block.without_terminator()) {
    if (isa<YieldOp, UnitOp>(&op))
      continue;
    allOps.push_back(&op);
    if (std::optional<int64_t> stage = getStage(&op))
      stageOps[*stage].push_back(&op);
  }

  if (stageOps.empty())
    return success();

  // Map each op to its assigned stage.
  DenseMap<Operation *, int64_t> opStage;
  for (auto &[stage, ops] : stageOps)
    for (Operation *op : ops)
      opStage[op] = stage;

  // Assign unannotated ops to the earliest stage that uses their results.
  for (Operation *op : allOps) {
    if (opStage.count(op))
      continue;
    std::optional<int64_t> assigned;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user->getBlock() != &block)
          continue;
        auto it = opStage.find(user);
        if (it == opStage.end())
          continue;
        if (!assigned || it->second < *assigned)
          assigned = it->second;
      }
    }
    if (!assigned) {
      op->emitError("op has no sched.stage attr and is not an SSA dep of any "
                    "staged op; cannot assign to a unit");
      return failure();
    }
    opStage[op] = *assigned;
  }

  // Verify dominance: for each op, all defs it depends on should be in an
  // earlier or equal stage (in program order).
  for (Operation *op : allOps) {
    int64_t myStage = opStage[op];
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || def->getBlock() != &block)
        continue;
      auto it = opStage.find(def);
      if (it == opStage.end())
        continue;
      if (it->second > myStage) {
        op->emitError("grouping into sched.unit ops would violate dominance: "
                      "op in stage ")
            << myStage << " uses a value defined in stage " << it->second;
        return failure();
      }
    }
  }

  // Build the sorted list of stages.
  SmallVector<int64_t> sortedStages;
  for (auto &[stage, _] : stageOps)
    sortedStages.push_back(stage);
  llvm::sort(sortedStages);

  // For each stage, wrap the assigned ops into a sched.unit op.
  for (int64_t stage : sortedStages) {
    // Collect ops for this stage in program order.
    SmallVector<Operation *> unitOps;
    for (Operation *op : allOps)
      if (opStage[op] == stage)
        unitOps.push_back(op);

    if (unitOps.empty())
      continue;

    // Collect inputs (values defined outside the unit used inside).
    // Collect outputs (values defined inside the unit used outside).
    llvm::SetVector<Value> inputs;
    llvm::SetVector<Value> outputs;
    llvm::SetVector<Operation *> unitOpSet(unitOps.begin(), unitOps.end());

    for (Operation *op : unitOps) {
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || !unitOpSet.count(def))
          inputs.insert(operand);
      }
      for (Value result : op->getResults()) {
        bool usedOutside =
            llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !unitOpSet.count(user);
            });
        if (usedOutside)
          outputs.insert(result);
      }
    }

    Location loc = unitOps.front()->getLoc();

    // Create the sched.unit op before the first op of this stage.
    builder.setInsertionPoint(unitOps.front());

    SmallVector<Value> inputVals(inputs.begin(), inputs.end());
    SmallVector<Type> resultTypes;
    for (Value v : outputs)
      resultTypes.push_back(v.getType());

    IntegerAttr stageAttr =
        builder.getIntegerAttr(builder.getIndexType(), stage);
    UnitOp unitOp = builder.create<UnitOp>(loc, resultTypes, inputVals,
                                           stageAttr, /*barrier=*/UnitAttr{});

    // Create the entry block with block arguments for each input.
    Block *unitBlock = builder.createBlock(&unitOp.getBody());
    for (Value input : inputVals)
      unitBlock->addArgument(input.getType(), loc);

    // Map each input value to its corresponding block argument.
    IRMapping mapping;
    for (auto [input, arg] : llvm::zip(inputVals, unitBlock->getArguments()))
      mapping.map(input, arg);

    // Clone ops into the unit block.
    builder.setInsertionPointToEnd(unitBlock);
    for (Operation *op : unitOps)
      builder.clone(*op, mapping);

    // Emit sched.yield with the output values (mapped to cloned results).
    SmallVector<Value> yieldOperands;
    for (Value v : outputs)
      yieldOperands.push_back(mapping.lookupOrDefault(v));
    builder.create<YieldOp>(loc, yieldOperands);

    // Replace external uses of outputs with the unit's results.
    for (auto &&[output, result] : llvm::zip(outputs, unitOp.getResults())) {
      Value out = output;
      out.replaceUsesWithIf(result, [&](OpOperand &use) {
        return !unitOpSet.count(use.getOwner());
      });
    }
    // Erase the original ops (in reverse order to avoid use-before-def issues).
    for (Operation *op : llvm::reverse(unitOps))
      op->erase();
  }

  return success();
}

struct GroupStagesToUnitsPass
    : public mlir::aster::sched::impl::GroupStagesToUnitsBase<
          GroupStagesToUnitsPass> {
  using GroupStagesToUnitsBase::GroupStagesToUnitsBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    OpBuilder builder(root->getContext());
    bool hasFailed = false;
    root->walk([&](Operation *op) {
      if (hasFailed)
        return;
      if (!isa<LoopLikeOpInterface>(op))
        return;
      for (Region &region : op->getRegions())
        for (Block &block : region)
          if (failed(processBlock(block, builder))) {
            hasFailed = true;
            return;
          }
    });
    if (hasFailed)
      signalPassFailure();
  }
};

} // namespace
