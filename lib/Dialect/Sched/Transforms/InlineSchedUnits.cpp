//===- InlineSchedUnits.cpp - Inline sched.unit bodies ----------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inlines the body of every sched.unit op into its parent block.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Sched/IR/SchedOps.h"
#include "aster/Dialect/Sched/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster::sched {
#define GEN_PASS_DEF_INLINESCHEDUNITS
#include "aster/Dialect/Sched/Transforms/Passes.h.inc"
} // namespace mlir::aster::sched

using namespace mlir;
using namespace mlir::aster::sched;

namespace {

struct InlineSchedUnitsPass
    : public mlir::aster::sched::impl::InlineSchedUnitsBase<
          InlineSchedUnitsPass> {
  using InlineSchedUnitsBase::InlineSchedUnitsBase;

  void runOnOperation() override {
    SmallVector<UnitOp> units;
    getOperation()->walk([&](UnitOp op) { units.push_back(op); });

    for (UnitOp unit : units)
      inlineUnit(unit);
  }

  static void inlineUnit(UnitOp unit) {
    Block &unitBlock = unit.getBody().front();
    YieldOp yield = cast<YieldOp>(unitBlock.getTerminator());

    // Map block arguments (unit inputs) to the unit's operands.
    IRMapping mapping;
    for (auto [arg, operand] :
         llvm::zip(unitBlock.getArguments(), unit.getInputs()))
      mapping.map(arg, operand);

    // Clone all ops (except the yield) before the unit op.
    OpBuilder builder(unit);
    for (Operation &op : unitBlock.without_terminator())
      builder.clone(op, mapping);

    // Replace unit results with the (mapped) yield operands.
    for (auto [result, yieldVal] :
         llvm::zip(unit.getResults(), yield.getValues()))
      result.replaceAllUsesWith(mapping.lookupOrDefault(yieldVal));

    unit.erase();
  }
};

} // namespace
