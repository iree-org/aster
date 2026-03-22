//===- Passes.td - Sched passes ----------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_SCHED_TRANSFORMS_PASSES_TD
#define ASTER_DIALECT_SCHED_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Sched passes
//===----------------------------------------------------------------------===//

def ResourceExpansion : Pass<"sched-expand-resources"> {
  let summary = "Expand sched.loop_resource ops out of loops";
  let description = [{
    This pass moves `sched.loop_resource` operations out of their enclosing
    loop-like operations and expands each region:

    - `allocate` and `forward` regions are inlined before the loop to seed the
      loop's iter_args.
    - `deallocate` region is inlined after the loop.
    - `fence` region is inlined at the end of the loop body.

    If `num-its > 0`, the resource is duplicated `num-its` times (circular
    buffer) and the forwarded values are rotated across iterations.
  }];
  let options = [
    Option<"numIts", "num-its", "int", "0",
           "If > 0, duplicate the resource this many times and rotate the "
           "forwarded values">
  ];
}

def GroupStagesToUnits : Pass<"sched-group-stages-to-units"> {
  let summary = "Group loop body ops by stage into sched.unit ops";
  let description = [{
    This pass groups operations inside loop bodies by their `sched.stage`
    discardable integer attribute and wraps each group into a `sched.unit` op.

    All ops inside the loop body must end up inside exactly one `sched.unit`.
    Unannotated ops that are SSA dependencies of staged ops are assigned to the
    earliest (lowest stage number) stage that consumes their results. If an
    unannotated op is consumed by multiple stages, it is assigned to the
    dominating one.

    The pass verifies that:
    - Each stage number appears at most once per loop body.
    - The resulting unit ordering satisfies SSA dominance.

    The pass fails if either invariant is violated. The `sched.stage`
    discardable attributes are preserved on the ops inside the units.
  }];
  let dependentDialects = ["mlir::aster::sched::SchedDialect"];
}

def InlineSchedUnits : Pass<"sched-inline-units"> {
  let summary = "Inline sched.unit bodies into their parent block";
  let description = [{
    This pass inlines the body of every `sched.unit` operation directly into
    the enclosing block, replacing the unit's results with the values yielded
    by its `sched.yield` terminator.
  }];
}

def LoopPipeliner : Pass<"sched-pipeline-loop"> {
  let summary = "Pipeline a loop around a target stage";
  let description = [{
    This pass pipelines a loop that contains `sched.unit` ops with `stage`
    attributes. Given a target stage `target-stage`, it restructures the loop
    into three parts using the SSA dependency graph (built via
    `aster_utils.ssa_scheduler`):

    - Prologue: all stages that `target-stage` depends on, plus `target-stage`
      itself, executed for the first iteration.
    - Loop body: the stages that come after `target-stage` (epilogue stages)
      followed by the prologue stages and `target-stage` for the next iteration.
    - Epilogue: the stages that come after `target-stage`, executed for the
      last iteration.

    Cross-iteration values are threaded through loop `iter_args`.
  }];
  let options = [
    Option<"tgtStage", "target-stage", "int", "0",
           "The stage number to use as the pipeline target">
  ];
  let dependentDialects = ["mlir::aster::aster_utils::AsterUtilsDialect"];
}

#endif // ASTER_DIALECT_SCHED_TRANSFORMS_PASSES_TD
