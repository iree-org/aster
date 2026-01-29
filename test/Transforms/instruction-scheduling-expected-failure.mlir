// RUN: aster-opt --aster-op-scheduling="test-only=true" --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// Test that pass fails gracefully with a warning when scheduling violates SSA chains.

// CHECK: warning: op scheduling: 'test.consumer_permuted' depends on
// CHECK-SAME: produced by 'test.producer_1' which hasn't been cloned yet
// CHECK-SAME: This indicates a scheduling violation
// CHECK: warning: op scheduling: failed to materialize operation schedule
// CHECK-LABEL: func.func @test_ssa_chains_invalid()
func.func @test_ssa_chains_invalid() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // All ops have same delay=0, rate=1, but consumer has permutation that
  // causes it to be scheduled for a different iteration than its producer,
  // violating SSA chains.
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
    %1 = "test.producer_2"(%0) {sched.delay = 0 : i32, sched.rate = 1 : i32} : (i32) -> f32
    %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  } {sched.dims = array<i64: 2, 2, 2>}
  return
}
