// RUN: aster-opt --aster-op-scheduling="num-iterations=1 dims=2,2,2 test-only=true" --allow-unregistered-dialect %s | FileCheck %s

// Test that pass fails gracefully with a warning when scheduling violates SSA chains.

// CHECK-LABEL: func.func @test_ssa_chains_invalid()
//      CHECK:   %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
// CHECK-NEXT:   %1 = "test.producer_2"(%0) {sched.delay = 0 : i32, sched.rate = 1 : i32} : (i32) -> f32
// CHECK-NEXT:   %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 0 : i32, sched.permutation = array<i32: 2, 0, 1>, sched.rate = 1 : i32} : (i32, f32) -> i8
// CHECK-NEXT:   return
func.func @test_ssa_chains_invalid() {
  %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
  %1 = "test.producer_2"(%0) {sched.delay = 0 : i32, sched.rate = 1 : i32} : (i32) -> f32
  %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  return
}

// CHECK-LABEL: func.func @test_ssa_chains_with_permutation()
//      CHECK:   %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
// CHECK-NEXT:   %1 = "test.producer_2"(%0) {sched.delay = 2 : i32, sched.rate = 2 : i32} : (i32) -> f32
// CHECK-NEXT:   %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 3 : i32, sched.permutation = array<i32: 2, 0, 1>, sched.rate = 3 : i32} : (i32, f32) -> i8
// CHECK-NEXT:   return
func.func @test_ssa_chains_with_permutation() {
  %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
  %1 = "test.producer_2"(%0) {sched.delay = 2 : i32, sched.rate = 2 : i32} : (i32) -> f32
  %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 3 : i32, sched.rate = 3 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  return
}
