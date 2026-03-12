// RUN: aster-opt %s --aster-apply-sched=scheds=sched -allow-unregistered-dialect | FileCheck %s

#sched = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #aster_utils.benefit_sched>

// BenefitSched selects the node with the highest label when multiple are ready.
// This is the opposite of StageTopoSortSched which selects the smallest label.

// CHECK-LABEL:   func.func @benefit_flat_block() -> i32 {
// Benefit order: among c1(stage 2), c2(stage 0), c3(stage 1) - pick c1 first (highest)
// Then c3, then c2, then addi(stage 3), then muli(stage 4)
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 2 : i32} 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant {sched.stage = 1 : i32} 3 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_2]], %[[CONSTANT_1]] {sched.stage = 3 : i32} : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_0]], %[[ADDI_0]] {sched.stage = 4 : i32} : i32
// CHECK:           return %[[MULI_0]] : i32
// CHECK:         }
func.func @benefit_flat_block() -> i32 attributes {sched = #sched} {
  %c1 = arith.constant {sched.stage = 2 : i32} 1 : i32
  %c2 = arith.constant {sched.stage = 0 : i32} 2 : i32
  %c3 = arith.constant {sched.stage = 1 : i32} 3 : i32
  %a = arith.addi %c2, %c3 {sched.stage = 3 : i32} : i32
  %b = arith.muli %c1, %a {sched.stage = 4 : i32} : i32
  return %b : i32
}

// CHECK-LABEL:   func.func @benefit_ssa_scheduler() -> i32 {
// Benefit scheduling: when ready, pick highest label. Higher stages are
// scheduled earlier when their dependencies are satisfied.
// CHECK:           %[[VAL_0:.*]] = "test.inst"() {sched.stage = 2 : i32} : () -> i32
// CHECK:           %[[VAL_1:.*]] = "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           %[[VAL_2:.*]] = "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           %[[VAL_3:.*]] = "test.inst"(%[[VAL_0]], %[[VAL_2]]) {sched.stage = 2 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_4:.*]] = "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[VAL_5:.*]] = "test.inst"(%[[VAL_4]], %[[VAL_1]]) {sched.stage = 4 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[VAL_7:.*]] = "test.inst"(%[[VAL_6]], %[[VAL_3]]) {sched.stage = 0 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_8:.*]] = "test.inst"(%[[VAL_5]], %[[VAL_7]]) {sched.stage = 5 : i32} : (i32, i32) -> i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }
func.func @benefit_ssa_scheduler() -> i32 attributes {sched = #sched} {
  %0 = "test.inst"() {sched.stage = 0 : i32} : () -> i32
  %1 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  %2 = "test.inst"() {sched.stage = 0 : i32} : () -> i32
  %3 = "test.inst"() {sched.stage = 2 : i32} : () -> i32
  %4 = "test.inst"() {sched.stage = 1 : i32} : () -> i32
  %5 = "test.inst"(%0, %1) {sched.stage = 4 : i32} : (i32, i32) -> i32
  %6 = "test.inst"(%3, %4) {sched.stage = 2 : i32} : (i32, i32) -> i32
  %7 = "test.inst"(%2, %6) {sched.stage = 0 : i32} : (i32, i32) -> i32
  %8 = "test.inst"(%5, %7) {sched.stage = 5 : i32} : (i32, i32) -> i32
  return %8 : i32
}

// Test tie-breaking: when labels are equal, smaller node ID wins
// CHECK-LABEL:   func.func @benefit_tie_break() -> i32 {
// All three constants have stage 0. With benefit_sched, when labels tie,
// we pick smaller node ID. Order: first constant (node 0), second (node 1),
// third (node 2), then addi (depends on all three)
// CHECK:           %[[C0:.*]] = arith.constant {sched.stage = 0 : i32} 1 : i32
// CHECK:           %[[C1:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:           %[[C2:.*]] = arith.constant {sched.stage = 0 : i32} 3 : i32
// CHECK:           %[[ADD:.*]] = arith.addi %[[C0]], %[[C1]] {sched.stage = 0 : i32} : i32
// CHECK:           return %[[ADD]] : i32
// CHECK:         }
func.func @benefit_tie_break() -> i32 attributes {sched = #sched} {
  %c1 = arith.constant {sched.stage = 0 : i32} 1 : i32
  %c2 = arith.constant {sched.stage = 0 : i32} 2 : i32
  %c3 = arith.constant {sched.stage = 0 : i32} 3 : i32
  %add = arith.addi %c1, %c2 {sched.stage = 0 : i32} : i32
  return %add : i32
}

// Test that higher-label ops are preferred when dependencies allow
// CHECK-LABEL:   func.func @benefit_highest_first() -> i32 {
// Two independent chains: a(stage 0)->b(stage 2) and c(stage 1)->d(stage 3)
// Initially ready: a(0), c(1). Pick c (higher label).
// Then ready: a, d. Pick d (higher label).
// Then ready: a. Pick a.
// Then ready: b. Pick b.
// Order: c, d, a, b
// CHECK:           %[[C:.*]] = arith.constant {sched.stage = 1 : i32} 10 : i32
// CHECK:           %[[D:.*]] = arith.addi %[[C]], %[[C]] {sched.stage = 3 : i32} : i32
// CHECK:           %[[A:.*]] = arith.constant {sched.stage = 0 : i32} 5 : i32
// CHECK:           %[[B:.*]] = arith.addi %[[A]], %[[A]] {sched.stage = 2 : i32} : i32
// CHECK:           %[[SUM:.*]] = arith.addi %[[B]], %[[D]] {sched.stage = 4 : i32} : i32
// CHECK:           return %[[SUM]] : i32
// CHECK:         }
func.func @benefit_highest_first() -> i32 attributes {sched = #sched} {
  %a = arith.constant {sched.stage = 0 : i32} 5 : i32
  %b = arith.addi %a, %a {sched.stage = 2 : i32} : i32
  %c = arith.constant {sched.stage = 1 : i32} 10 : i32
  %d = arith.addi %c, %c {sched.stage = 3 : i32} : i32
  %sum = arith.addi %b, %d {sched.stage = 4 : i32} : i32
  return %sum : i32
}
