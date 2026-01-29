// RUN: aster-opt --aster-op-scheduling="test-only=true" --allow-unregistered-dialect -split-input-file %s 2>&1 | FileCheck %s

// Test that OpScheduling pass emits proper warnings for various edge cases.
// All warnings are emitted before module outputs, so check them in order first.

// CHECK: warning: op scheduling: skipping loop - loop yields values
// CHECK-SAME: Operation scheduling only supports loops that don't yield values atm
// CHECK: warning: op scheduling: skipping loop - trip count (8) does not match product of dimensions (4)
// CHECK-SAME: Ensure 'sched.dims' matches the actual loop iteration space
// CHECK: warning: op scheduling: skipping loop - no operations to schedule in loop body
// CHECK-SAME: only terminator found
// CHECK: warning: op scheduling: skipping loop - dynamic bounds detected
// CHECK-SAME: Operation scheduling requires constant trip count atm

// Now check function labels appear in output (in order after warnings)
// CHECK-LABEL: func.func @test_loop_with_yields
// CHECK-LABEL: func.func @test_trip_count_mismatch
// CHECK-LABEL: func.func @test_empty_loop_body
// CHECK-LABEL: func.func @test_dynamic_bounds
// CHECK-LABEL: func.func @test_no_sched_dims

// -----
// Test warning for loop that yields values (not supported).
func.func @test_loop_with_yields() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init = arith.constant 0.0 : f32
  // Loop yields a value - should be skipped with warning
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
    %new_acc = arith.addf %acc, %val : f32
    scf.yield %new_acc : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----
// Test warning for trip count mismatch with sched.dims.
func.func @test_trip_count_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // Trip count is 8, but sched.dims says 2x2=4 - mismatch!
  scf.for %i = %c0 to %c8 step %c1 {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
  } {sched.dims = array<i64: 2, 2>}
  return
}

// -----
// Test warning for empty loop body (only terminator).
func.func @test_empty_loop_body() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Empty body - only terminator (scf.yield) is present
  } {sched.dims = array<i64: 4>}
  return
}

// -----
// Test warning for dynamic loop bounds.
func.func @test_dynamic_bounds(%upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Dynamic upper bound - should be skipped with warning
  scf.for %i = %c0 to %upper step %c1 {
    %val = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0 : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----
// Test that loop without sched.dims is silently skipped (no warning).
func.func @test_no_sched_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // No sched.dims attribute - silently skipped
  scf.for %i = %c0 to %c4 step %c1 {
    %val = arith.constant 1.0 : f32
  }
  return
}
