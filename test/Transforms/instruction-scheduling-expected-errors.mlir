// RUN: aster-opt --aster-op-scheduling="test-only=true" --allow-unregistered-dialect --verify-diagnostics -split-input-file %s

// Test that OpScheduling pass emits proper errors for various failure cases.

// Test error for loop that yields values (not supported).
func.func @test_loop_with_yields() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0.0 : f32
  // expected-error @below {{op scheduling: loop yields values}}
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
    %new_acc = arith.addf %acc, %val : f32
    scf.yield %new_acc : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test error for trip count mismatch with sched.dims.
func.func @test_trip_count_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{op scheduling: trip count (8) does not match product of dimensions (4)}}
  scf.for %i = %c0 to %c8 step %c1 {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
  } {sched.dims = array<i64: 2, 2>}
  return
}

// -----

// Test error for empty loop body (only terminator).
func.func @test_empty_loop_body() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @below {{op scheduling: no operations to schedule in loop body}}
  scf.for %i = %c0 to %c4 step %c1 {
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test error for dynamic loop bounds.
func.func @test_dynamic_bounds(%upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{op scheduling: dynamic bounds detected}}
  scf.for %i = %c0 to %upper step %c1 {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test error for SSA chain violation with permutation.
func.func @test_ssa_chains_invalid() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{op scheduling: failed to materialize operation schedule}}
  scf.for %i = %c0 to %c8 step %c1 {
    %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
    %1 = "test.producer_2"(%0) {sched.delay = 0 : i32, sched.rate = 1 : i32} : (i32) -> f32
    // expected-error @below {{op scheduling: 'test.consumer_permuted' depends on}}
    %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  } {sched.dims = array<i64: 2, 2, 2>}
  return
}

// -----

// Test error when nested region uses value from operation scheduled later.
func.func @test_nested_region_uses_later_producer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true

  // expected-error @below {{op scheduling: failed to materialize operation schedule}}
  scf.for %i = %c0 to %c2 step %c1 {
    // Producer at delay=4 - defined first (valid SSA), but fires later due to delay
    %later_val = arith.muli %i, %c2 {sched.delay = 4 : i32, sched.rate = 1 : i32} : index

    // scf.if at delay=0 fires first in schedule, but uses %later_val
    scf.if %true {
      // expected-error @below {{op scheduling: operation inside nested region uses}}
      %use_it = arith.addi %later_val, %c1 : index
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}
