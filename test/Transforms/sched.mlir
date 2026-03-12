// RUN: aster-opt %s --aster-apply-sched=scheds=sched -allow-unregistered-dialect | FileCheck %s

#sched = #aster_utils.generic_scheduler<#aster_utils.ssa_scheduler, #aster_utils.sched_stage_labeler, #aster_utils.stage_topo_sort_sched>

// CHECK-LABEL:   func.func @test_ssa_scheduler() -> i32 {
// CHECK:           %[[VAL_0:.*]] = "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[VAL_1:.*]] = "test.inst"() {sched.stage = 0 : i32} : () -> i32
// CHECK:           %[[VAL_2:.*]] = "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           %[[VAL_3:.*]] = "test.inst"() {sched.stage = 1 : i32} : () -> i32
// CHECK:           %[[VAL_4:.*]] = "test.inst"() {sched.stage = 2 : i32} : () -> i32
// CHECK:           %[[VAL_5:.*]] = "test.inst"(%[[VAL_4]], %[[VAL_3]]) {sched.stage = 2 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = "test.inst"(%[[VAL_1]], %[[VAL_5]]) {sched.stage = 0 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_7:.*]] = "test.inst"(%[[VAL_0]], %[[VAL_2]]) {sched.stage = 4 : i32} : (i32, i32) -> i32
// CHECK:           %[[VAL_8:.*]] = "test.inst"(%[[VAL_7]], %[[VAL_6]]) {sched.stage = 5 : i32} : (i32, i32) -> i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }
func.func @test_ssa_scheduler() -> i32 attributes {sched = #sched} {
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

// CHECK-LABEL:   func.func @flat_block() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant {sched.stage = 1 : i32} 3 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant {sched.stage = 2 : i32} 1 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_1]] {sched.stage = 3 : i32} : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[ADDI_0]] {sched.stage = 4 : i32} : i32
// CHECK:           return %[[MULI_0]] : i32
// CHECK:         }
func.func @flat_block() -> i32 attributes {sched = #sched} {
  %c1 = arith.constant {sched.stage = 2 : i32} 1 : i32
  %c2 = arith.constant {sched.stage = 0 : i32} 2 : i32
  %c3 = arith.constant {sched.stage = 1 : i32} 3 : i32
  %a = arith.addi %c2, %c3 {sched.stage = 3 : i32} : i32
  %b = arith.muli %c1, %a {sched.stage = 4 : i32} : i32
  return %b : i32
}

// CHECK-LABEL:   func.func @loop_with_iter_args(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 0 : i32} 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant {sched.stage = 0 : i32} 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant {sched.stage = 1 : i32} 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[ARG0]]) -> (i32) {
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant {sched.stage = 1 : i32} 10 : i32
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_2]] {sched.stage = 2 : i32} : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_4]] {sched.stage = 3 : i32} : i32
// CHECK:             %[[MINSI_0:.*]] = arith.minsi %[[MULI_0]], %[[CONSTANT_5]] {sched.stage = 4 : i32} : i32
// CHECK:             scf.yield %[[MINSI_0]] : i32
// CHECK:           }
// CHECK:           return %[[FOR_0]] : i32
// CHECK:         }
func.func @loop_with_iter_args(%arg0: i32) -> i32 attributes {sched = #sched} {
  %c0 = arith.constant {sched.stage = 0 : i32} 0 : index
  %c4 = arith.constant {sched.stage = 0 : i32} 4 : index
  %c1 = arith.constant {sched.stage = 1 : i32} 1 : i32
  %c1_idx = arith.constant 1 : index
  %sum = scf.for %i = %c0 to %c4 step %c1_idx iter_args(%acc = %arg0) -> (i32) {
    %inc = arith.addi %acc, %c1 {sched.stage = 2 : i32} : i32
    %c10 = arith.constant {sched.stage = 1 : i32} 10 : i32
    %c2 = arith.constant {sched.stage = 0 : i32} 2 : i32
    %scaled = arith.muli %inc, %c2 {sched.stage = 3 : i32} : i32
    %capped = arith.minsi %scaled, %c10 {sched.stage = 4 : i32} : i32
    scf.yield %capped : i32
  } {sched = #sched}
  return %sum : i32
}

// CHECK-LABEL:   func.func @multiple_top_level_loops(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 0 : i32} 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant {sched.stage = 0 : i32} 2 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant {sched.stage = 0 : i32} 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant {sched.stage = 1 : i32} 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] iter_args(%[[VAL_1:.*]] = %[[ARG0]]) -> (i32) {
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant {sched.stage = 0 : i32} 3 : i32
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_3]] {sched.stage = 1 : i32} : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_5]] {sched.stage = 2 : i32} : i32
// CHECK:             scf.yield %[[MULI_0]] : i32
// CHECK:           }
// CHECK:           %[[FOR_1:.*]] = scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] iter_args(%[[VAL_3:.*]] = %[[FOR_0]]) -> (i32) {
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_3]] {sched.stage = 1 : i32} : i32
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_6]] {sched.stage = 2 : i32} : i32
// CHECK:             scf.yield %[[DIVSI_0]] : i32
// CHECK:           }
// CHECK:           return %[[FOR_1]] : i32
// CHECK:         }
func.func @multiple_top_level_loops(%arg0: i32) -> i32 attributes {sched = #sched} {
  %c0 = arith.constant {sched.stage = 0 : i32} 0 : index
  %c2 = arith.constant {sched.stage = 0 : i32} 2 : index
  %c1_idx = arith.constant 1 : index
  %c1 = arith.constant {sched.stage = 1 : i32} 1 : i32
  %first = scf.for %i = %c0 to %c2 step %c1_idx iter_args(%a = %arg0) -> (i32) {
    %x = arith.addi %a, %c1 {sched.stage = 1 : i32} : i32
    %c3 = arith.constant {sched.stage = 0 : i32} 3 : i32
    %y = arith.muli %x, %c3 {sched.stage = 2 : i32} : i32
    scf.yield %y : i32
  } {sched = #sched}
  %c4 = arith.constant {sched.stage = 0 : i32} 4 : index
  %second = scf.for %j = %c0 to %c4 step %c1_idx iter_args(%b = %first) -> (i32) {
    %dec = arith.subi %b, %c1 {sched.stage = 1 : i32} : i32
    %c2_i32 = arith.constant {sched.stage = 0 : i32} 2 : i32
    %halved = arith.divsi %dec, %c2_i32 {sched.stage = 2 : i32} : i32
    scf.yield %halved : i32
  } {sched = #sched}
  return %second : i32
}

// CHECK-LABEL:   func.func @multiple_independent_loops(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 0 : i32} 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant {sched.stage = 0 : i32} 2 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant {sched.stage = 0 : i32} 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant {sched.stage = 1 : i32} 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] iter_args(%[[VAL_1:.*]] = %[[ARG1]]) -> (i32) {
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant {sched.stage = 0 : i32} 2 : i32
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_3]] {sched.stage = 1 : i32} : i32
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_5]] {sched.stage = 2 : i32} : i32
// CHECK:             scf.yield %[[DIVSI_0]] : i32
// CHECK:           } {sched.stage = 3 : i32}
// CHECK:           %[[FOR_1:.*]] = scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] iter_args(%[[VAL_3:.*]] = %[[ARG0]]) -> (i32) {
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant {sched.stage = 0 : i32} 3 : i32
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_3]] {sched.stage = 1 : i32} : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_6]] {sched.stage = 2 : i32} : i32
// CHECK:             scf.yield %[[MULI_0]] : i32
// CHECK:           } {sched.stage = 4 : i32}
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[FOR_1]], %[[FOR_0]] {sched.stage = 5 : i32} : i32
// CHECK:           return %[[ADDI_1]] : i32
// CHECK:         }
func.func @multiple_independent_loops(%arg0: i32, %arg1: i32) -> i32 attributes {sched = #sched} {
  %c0 = arith.constant {sched.stage = 0 : i32} 0 : index
  %c2 = arith.constant {sched.stage = 0 : i32} 2 : index
  %c1_idx = arith.constant 1 : index
  %c1 = arith.constant {sched.stage = 1 : i32} 1 : i32
  %first = scf.for %i = %c0 to %c2 step %c1_idx iter_args(%a = %arg0) -> (i32) {
    %x = arith.addi %a, %c1 {sched.stage = 1 : i32} : i32
    %c3 = arith.constant {sched.stage = 0 : i32} 3 : i32
    %y = arith.muli %x, %c3 {sched.stage = 2 : i32} : i32
    scf.yield %y : i32
  } {sched = #sched, sched.stage = 4 : i32}
  %c4 = arith.constant {sched.stage = 0 : i32} 4 : index
  %second = scf.for %j = %c0 to %c4 step %c1_idx iter_args(%b = %arg1) -> (i32) {
    %dec = arith.subi %b, %c1 {sched.stage = 1 : i32} : i32
    %c2_i32 = arith.constant {sched.stage = 0 : i32} 2 : i32
    %halved = arith.divsi %dec, %c2_i32 {sched.stage = 2 : i32} : i32
    scf.yield %halved : i32
  } {sched = #sched, sched.stage = 3 : i32}
  %sum = arith.addi %first, %second {sched.stage = 5 : i32} : i32
  return %sum : i32
}
