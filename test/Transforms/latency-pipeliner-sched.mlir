// RUN: aster-opt %s --aster-apply-sched=scheds=sched -allow-unregistered-dialect | FileCheck %s

// Scheduler definitions used across tests.
//   sched1 - schedLimit=1 (default): at most one op selected per invocation.
//   sched2 - schedLimit=2: at most two ops selected per invocation.
//   sched3 - schedLimit=3: at most three ops selected per invocation.
//   sched0 - schedLimit=0 (unlimited): all ready ops selected per invocation.
#sched1 = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #amdgcn.latency_pipeliner_sched<1>>
#sched2 = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #amdgcn.latency_pipeliner_sched<2>>
#sched3 = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #amdgcn.latency_pipeliner_sched<3>>
#sched0 = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #amdgcn.latency_pipeliner_sched<0>>

// Phase 1: a single label-0 op is scheduled immediately, before any higher-label
// ops, regardless of its position in the source block.
//
// Source order : n0(3), n1(2), n2(0), n3(1) — all independent.
// Expected order: n2(0) first (pinned), then n0(3), n1(2), n3(1) by descending label.
//
// CHECK-LABEL: func.func @phase1_zero_label_immediate
// CHECK:           %[[N2:.*]] = "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[N0:.*]] = "test.inst"() {sched.stage = 3 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 2 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           return %[[N0]] : i32
func.func @phase1_zero_label_immediate() -> i32 attributes {sched = #sched1} {
  %0 = "test.inst"() {sched.stage = 3 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 2 : i32} : () -> i32
  %2 = "test.inst"() {sched.stage = 0 : i32} : () -> i32
  %3 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  return %0 : i32
}

// Phase 1: all label-0 ops in the ready set are scheduled in a single invocation,
// independently of schedLimit.
//
// Source order: n0(0), n1(2), n2(0), n3(1).
// Expected: n0 and n2 (both label 0) come first in one batch, then n1(2), n3(1).
//
// CHECK-LABEL: func.func @phase1_multiple_zero_labels
// CHECK:           "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[N1:.*]] = "test.inst"() {sched.stage = 2 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           return %[[N1]] : i32
func.func @phase1_multiple_zero_labels() -> i32 attributes {sched = #sched1} {
  %0 = "test.inst"() {sched.stage = 0 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 2 : i32} : () -> i32
  %2 = "test.inst"() {sched.stage = 0 : i32} : () -> i32
  %3 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  return %1 : i32
}

// Phase 2: with schedLimit=2 the scheduler interleaves ops from different label
// groups.  At each invocation the highest-label op is picked first, then one op
// from a different label group.
//
// Source order: n0(3), n1(1), n2(3), n3(1) — all independent.
// Invocation 1: sorted ready = [n0(3), n2(3), n1(1), n3(1)].
//   Pick n0 (highest, label 3), then n1 (different label 1). → schedule n0, n1.
// Invocation 2: ready = [n2(3), n3(1)].
//   Pick n2, then n3. → schedule n2, n3.
// Expected order: n0(3), n1(1), n2(3), n3(1) — interleaved.
//
// CHECK-LABEL: func.func @phase2_interleaving
// CHECK:           %[[N0:.*]] = "test.inst"() {sched.stage = 3 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 3 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           return %[[N0]] : i32
func.func @phase2_interleaving() -> i32 attributes {sched = #sched2} {
  %0 = "test.inst"() {sched.stage = 3 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  %2 = "test.inst"() {sched.stage = 3 : i32} : () -> i32
  %3 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  return %0 : i32
}

// Phase 2: when all ready ops share the same label the scheduler must still
// schedule all of them without starvation (regression test for the same-label
// bug where only the first op would ever be inserted).
//
// Source / expected order: c1, c2, c3 — unchanged (same label, ascending node ID).
//
// CHECK-LABEL: func.func @phase2_same_labels
// CHECK:           %[[C1:.*]] = arith.constant {sched.stage = 2 : i32} 1 : i32
// CHECK:           arith.constant {sched.stage = 2 : i32} 2 : i32
// CHECK:           arith.constant {sched.stage = 2 : i32} 3 : i32
// CHECK:           return %[[C1]] : i32
func.func @phase2_same_labels() -> i32 attributes {sched = #sched3} {
  %c1 = arith.constant {sched.stage = 2 : i32} 1 : i32
  %c2 = arith.constant {sched.stage = 2 : i32} 2 : i32
  %c3 = arith.constant {sched.stage = 2 : i32} 3 : i32
  return %c1 : i32
}

// schedLimit=0 (non-positive): all ready ops are selected in a single invocation.
// The order within the invocation follows the interleaving rule (descending label,
// different-label preference).
//
// Source order: n0(3), n1(1), n2(2), n3(1) — all independent.
// All 4 are ready. sorted = [n0(3), n2(2), n1(1), n3(1)].
//   Insert n0(3), then n2(2, different), then n1(1, different), then n3(1,
//   same as n1 but picked after reset) → schedule n0, n2, n1, n3.
//
// CHECK-LABEL: func.func @schedlimit_unlimited
// CHECK:           %[[N0:.*]] = "test.inst"() {sched.stage = 3 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 2 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           return %[[N0]] : i32
func.func @schedlimit_unlimited() -> i32 attributes {sched = #sched0} {
  %0 = "test.inst"() {sched.stage = 3 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  %2 = "test.inst"() {sched.stage = 2 : i32} : () -> i32
  %3 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  return %0 : i32
}

// SSA dependencies are always respected: an op cannot be scheduled before its
// operands, even if its label would otherwise give it priority.
//
// n2 depends on both n0 and n1. With schedLimit=2:
//   Invocation 1: ready = [n0(1), n1(3)]. sorted = [n1(3), n0(1)].
//     Pick n1 (highest), then n0 (different label). → schedule n1, n0.
//   Invocation 2: n2 is now ready. → schedule n2.
//
// CHECK-LABEL: func.func @respects_ssa_deps
// CHECK:           %[[N1:.*]] = "test.inst"() {sched.stage = 3 : i32} : () -> i32
// CHECK:           %[[N0:.*]] = "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           %[[N2:.*]] = "test.inst"(%[[N0]], %[[N1]]) {sched.stage = 2 : i32} : (i32, i32) -> i32
// CHECK:           return %[[N2]] : i32
func.func @respects_ssa_deps() -> i32 attributes {sched = #sched2} {
  %0 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 3 : i32} : () -> i32
  %2 = "test.inst"(%0, %1) {sched.stage = 2 : i32} : (i32, i32) -> i32
  return %2 : i32
}
