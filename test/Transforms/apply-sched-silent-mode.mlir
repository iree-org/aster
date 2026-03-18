// Tests for the silent-mode option of aster-apply-sched.
//
// When silent-mode=true (default), a missing schedule does not cause a
// failure. When no scheds are specified, "aster.sched" is used as default.

// RUN: aster-opt %s -split-input-file \
// RUN:   --aster-apply-sched="scheds=missing silent-mode=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s --check-prefix=SILENT

// RUN: aster-opt %s -split-input-file \
// RUN:   --aster-apply-sched -allow-unregistered-dialect \
// RUN:   | FileCheck %s --check-prefix=DEFAULT

// Verify that when silent-mode=true and the requested schedule is absent, the
// pass succeeds and the function is passed through unchanged.

// SILENT-LABEL: func.func @silent_missing_sched
// SILENT-NEXT:    return
func.func @silent_missing_sched() {
  return
}

// -----

#sched = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #aster_utils.stage_topo_sort_sched>

// Verify that when no scheds are specified, the default "aster.sched" name is
// used and the schedule is applied (operations are reordered by stage).

// DEFAULT-LABEL: func.func @default_sched_name
// DEFAULT-NEXT:    %[[C0:.*]] = arith.constant {sched.stage = 0 : i32} 0 : i32
// DEFAULT-NEXT:    %[[C1:.*]] = arith.constant {sched.stage = 1 : i32} 1 : i32
// DEFAULT-NEXT:    return %[[C0]], %[[C1]]
// SILENT-LABEL: func.func @default_sched_name
func.func @default_sched_name() -> (i32, i32)
    attributes {"aster.sched" = #sched} {
  %c1 = arith.constant {sched.stage = 1 : i32} 1 : i32
  %c0 = arith.constant {sched.stage = 0 : i32} 0 : i32
  return %c0, %c1 : i32, i32
}
