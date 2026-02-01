// RUN: aster-opt %s -amdgcn-instruction-scheduling-autoschedule | FileCheck %s

// Test that autoschedule pass handles operations with nested regions (scf.if).
// Phase 1: All top-level ops get scheduled (including ops with regions)
// Phase 2: Schedule is propagated to nested ops inside ops with regions

// Test basic scf.if - parent schedule propagates to nested ops
// CHECK-LABEL: func.func @test_scf_if_schedule_propagation
func.func @test_scf_if_schedule_propagation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c4 step %c1 {
    // scf.if has explicit schedule
    // CHECK: scf.if {{.*}} {
    // CHECK:   arith.constant {sched.delay = 2 : i32, sched.rate = 1 : i32} 42 : i32
    // CHECK: } {sched.delay = 2 : i32, sched.rate = 1 : i32}
    scf.if %true {
      // Nested op without schedule - should inherit parent's schedule
      %x = arith.constant 42 : i32
    } {sched.delay = 2 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 4>}

  return
}

// Test scf.if inherits schedule from consumer (Rule 1)
// CHECK-LABEL: func.func @test_scf_if_inherits_from_consumer
func.func @test_scf_if_inherits_from_consumer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c4 step %c1 {
    // scf.if without explicit schedule - should inherit from consumer
    // CHECK: scf.if {{.*}} -> (i32) {
    // CHECK:   arith.constant {sched.delay = 5 : i32, sched.rate = 2 : i32} 42 : i32
    // CHECK: } {sched.delay = 5 : i32, sched.rate = 2 : i32}
    %result = scf.if %true -> (i32) {
      %x = arith.constant 42 : i32
      scf.yield %x : i32
    } else {
      %y = arith.constant 0 : i32
      scf.yield %y : i32
    }

    // Consumer with explicit schedule
    %use = arith.muli %result, %result {sched.delay = 5 : i32, sched.rate = 2 : i32} : i32
  } {sched.dims = array<i64: 4>}

  return
}

// Test scf.if gets default schedule when no consumer (Rule 2)
// CHECK-LABEL: func.func @test_scf_if_default_schedule
func.func @test_scf_if_default_schedule() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c4 step %c1 {
    // scf.if without explicit schedule and no consumer - gets default
    // CHECK: scf.if {{.*}} {
    // CHECK:   arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 42 : i32
    // CHECK: } {sched.delay = 0 : i32, sched.rate = 1 : i32}
    scf.if %true {
      %x = arith.constant 42 : i32
    }
  } {sched.dims = array<i64: 4>}

  return
}

// Test scf.if respects operand delay constraints
// CHECK-LABEL: func.func @test_scf_if_operand_delay_constraint
func.func @test_scf_if_operand_delay_constraint() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  scf.for %i = %c0 to %c4 step %c1 {
    // Producer with high delay
    %cond = arith.cmpi eq, %i, %c0 {sched.delay = 10 : i32, sched.rate = 1 : i32} : index

    // scf.if uses %cond - should inherit delay >= 10 from operand
    // CHECK: scf.if {{.*}} {
    // CHECK:   arith.constant {sched.delay = 10 : i32, sched.rate = 1 : i32} 1 : i32
    // CHECK: } {sched.delay = 10 : i32, sched.rate = 1 : i32}
    scf.if %cond {
      %x = arith.constant 1 : i32
    }
  } {sched.dims = array<i64: 4>}

  return
}

// Test nested scf.if with else branch
// CHECK-LABEL: func.func @test_scf_if_else_schedule_propagation
func.func @test_scf_if_else_schedule_propagation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: scf.if {{.*}} -> (i32) {
    // CHECK:   arith.constant {sched.delay = 3 : i32, sched.rate = 1 : i32} 1 : i32
    // CHECK: } else {
    // CHECK:   arith.constant {sched.delay = 3 : i32, sched.rate = 1 : i32} 2 : i32
    // CHECK: } {sched.delay = 3 : i32, sched.rate = 1 : i32}
    %result = scf.if %true -> (i32) {
      %x = arith.constant 1 : i32
      scf.yield %x : i32
    } else {
      %y = arith.constant 2 : i32
      scf.yield %y : i32
    } {sched.delay = 3 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test multiple scf.if with different schedules
// CHECK-LABEL: func.func @test_multiple_scf_if_different_schedules
func.func @test_multiple_scf_if_different_schedules() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  %false = arith.constant false

  scf.for %i = %c0 to %c2 step %c1 {
    // First scf.if with delay=1
    // CHECK: scf.if {{.*}} {
    // CHECK:   arith.constant {sched.delay = 1 : i32, sched.rate = 1 : i32} 10 : i32
    // CHECK: } {sched.delay = 1 : i32, sched.rate = 1 : i32}
    scf.if %true {
      %a = arith.constant 10 : i32
    } {sched.delay = 1 : i32, sched.rate = 1 : i32}

    // Second scf.if with delay=5
    // CHECK: scf.if {{.*}} {
    // CHECK:   arith.constant {sched.delay = 5 : i32, sched.rate = 2 : i32} 20 : i32
    // CHECK: } {sched.delay = 5 : i32, sched.rate = 2 : i32}
    scf.if %false {
      %b = arith.constant 20 : i32
    } {sched.delay = 5 : i32, sched.rate = 2 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test that existing schedules on nested ops are preserved
// CHECK-LABEL: func.func @test_preserve_existing_nested_schedule
func.func @test_preserve_existing_nested_schedule() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: scf.if {{.*}} {
    // Nested op with existing schedule - should be preserved
    // CHECK:   arith.constant {sched.delay = 99 : i32, sched.rate = 99 : i32} 42 : i32
    // Nested op without schedule - inherits from parent
    // CHECK:   arith.constant {sched.delay = 2 : i32, sched.rate = 1 : i32} 43 : i32
    // CHECK: } {sched.delay = 2 : i32, sched.rate = 1 : i32}
    scf.if %true {
      %x = arith.constant {sched.delay = 99 : i32, sched.rate = 99 : i32} 42 : i32
      %y = arith.constant 43 : i32
    } {sched.delay = 2 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test deeply nested ops in scf.if
// CHECK-LABEL: func.func @test_deeply_nested_propagation
func.func @test_deeply_nested_propagation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: scf.if {{.*}} {
    // All nested ops should inherit delay=4
    // CHECK:   %[[A:.*]] = arith.constant {sched.delay = 4 : i32, sched.rate = 1 : i32} 1 : i32
    // CHECK:   %[[B:.*]] = arith.constant {sched.delay = 4 : i32, sched.rate = 1 : i32} 2 : i32
    // CHECK:   arith.addi %[[A]], %[[B]] {sched.delay = 4 : i32, sched.rate = 1 : i32}
    // CHECK:   arith.muli {{.*}} {sched.delay = 4 : i32, sched.rate = 1 : i32}
    // CHECK: } {sched.delay = 4 : i32, sched.rate = 1 : i32}
    scf.if %true {
      %a = arith.constant 1 : i32
      %b = arith.constant 2 : i32
      %c = arith.addi %a, %b : i32
      %d = arith.muli %c, %a : i32
    } {sched.delay = 4 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}
