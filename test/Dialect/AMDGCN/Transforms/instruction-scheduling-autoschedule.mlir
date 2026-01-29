// RUN: aster-opt %s -amdgcn-instruction-scheduling-autoschedule | FileCheck %s

// Rule 1: Operations without schedules that have a single consumer
// should inherit the consumer's schedule
// CHECK-LABEL: func.func @test_autoschedule_from_single_consumer
func.func @test_autoschedule_from_single_consumer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: arith.constant {sched.delay = 2 : i32, sched.rate = 4 : i32} 1.0
    %val1 = arith.constant 1.0 : f32
    %add = arith.addf %val1, %val1 {sched.delay = 2 : i32, sched.rate = 4 : i32} : f32

    // CHECK: %[[VAL2:.*]] = arith.constant {sched.delay = 1 : i32, sched.rate = 2 : i32} 2.0
    %val2 = arith.constant 2.0 : f32
    %mul = arith.mulf %val2, %val2 {sched.delay = 1 : i32, sched.rate = 2 : i32} : f32
  } {sched.dims = array<i64: 16>}
  return
}


// Rule 2: Operations at the end with no scheduled successors should get default
// schedule (delay=0, rate=1)
// CHECK-LABEL: func.func @test_default_schedule
func.func @test_default_schedule() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %val4 = arith.constant {sched.delay = 1 : i32, sched.rate = 2 : i32} 5.0 : f32

    // CHECK: %[[VAL5:.*]] = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 6.0
    %val5 = arith.constant 6.0 : f32
    // CHECK: %[[VAL6:.*]] = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 7.0
    %val6 = arith.constant 7.0 : f32
    // CHECK: %[[RESULT:.*]] = arith.addf %[[VAL5]], %[[VAL6]] {sched.delay = 0 : i32, sched.rate = 1 : i32}
    %result = arith.addf %val5, %val6 : f32
  } {sched.dims = array<i64: 4>}
  return
}


// Test a mix of rules 1 and 2.
// CHECK-LABEL: func.func @test_mixed_autoschedule
func.func @test_mixed_autoschedule() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // Group 1: Will inherit from single consumer
    // CHECK: arith.constant {sched.delay = 2 : i32, sched.rate = 4 : i32} 1.0
    %a = arith.constant 1.0 : f32
    %b = arith.addf %a, %a {sched.delay = 2 : i32, sched.rate = 4 : i32} : f32

    // Group 2: Will inherit smallest schedule from any consumer.
    // CHECK: arith.constant {sched.delay = 3 : i32, sched.rate = 2 : i32} 2.0
    %c = arith.constant 2.0 : f32
    %foo = arith.addf %b, %b {sched.delay = 300 : i32, sched.rate = 200 : i32} : f32
    // CHECK: arith.constant {sched.delay = 4 : i32, sched.rate = 2 : i32} 3.0
    %d = arith.constant 3.0 : f32
    %f = arith.addf %c, %c {sched.delay = 3 : i32, sched.rate = 2 : i32} : f32
    %e = arith.mulf %c, %d {sched.delay = 4 : i32, sched.rate = 2 : i32} : f32

    // Group 3: Will get default schedule
    // CHECK: arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 4.0
    %g = arith.constant 4.0 : f32
    // CHECK: arith.mulf {{.*}} {sched.delay = 0 : i32, sched.rate = 1 : i32}
    %h = arith.mulf %g, %g : f32
  } {sched.dims = array<i64: 8>}
  return
}

// -----

// Test operand delay propagation: operations without consumers should inherit
// max delay from their operands (Rule 2 with operand constraints).
// CHECK-LABEL: func.func @test_operand_delay_propagation
func.func @test_operand_delay_propagation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Producer with high delay
    // CHECK: arith.constant {sched.delay = 100 : i32, sched.rate = 1 : i32} 1.0
    %producer = arith.constant {sched.delay = 100 : i32, sched.rate = 1 : i32} 1.0 : f32

    // Consumer without explicit schedule - should inherit delay >= 100 from operand
    // CHECK: arith.addf {{.*}} {sched.delay = 100 : i32, sched.rate = 1 : i32}
    %consumer = arith.addf %producer, %producer : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test operand delay propagation with multiple operands: should take max delay
// CHECK-LABEL: func.func @test_max_operand_delay
func.func @test_max_operand_delay() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Two producers with different delays
    // CHECK: arith.constant {sched.delay = 50 : i32, sched.rate = 1 : i32} 1.0
    %prod1 = arith.constant {sched.delay = 50 : i32, sched.rate = 1 : i32} 1.0 : f32
    // CHECK: arith.constant {sched.delay = 150 : i32, sched.rate = 1 : i32} 2.0
    %prod2 = arith.constant {sched.delay = 150 : i32, sched.rate = 1 : i32} 2.0 : f32

    // Consumer without explicit schedule - should inherit max(50, 150) = 150
    // CHECK: arith.addf {{.*}} {sched.delay = 150 : i32, sched.rate = 1 : i32}
    %consumer = arith.addf %prod1, %prod2 : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test chained operand delay propagation with explicit consumer schedule:
// intermediate inherits from consumer, which must be >= operand delay
// CHECK-LABEL: func.func @test_chained_operand_delay
func.func @test_chained_operand_delay() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Root producer with high delay
    // CHECK: arith.constant {sched.delay = 200 : i32, sched.rate = 1 : i32} 1.0
    %root = arith.constant {sched.delay = 200 : i32, sched.rate = 1 : i32} 1.0 : f32

    // Intermediate without explicit schedule - inherits from consumer (delay=200)
    // which satisfies operand constraint (delay >= 200)
    // CHECK: arith.negf {{.*}} {sched.delay = 200 : i32, sched.rate = 2 : i32}
    %mid = arith.negf %root : f32

    // Explicit consumer with delay >= operand delay (200), so no conflict
    %final = arith.addf %mid, %mid {sched.delay = 200 : i32, sched.rate = 2 : i32} : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test that consumer inheritance still works when consumer delay >= operand delay
// CHECK-LABEL: func.func @test_consumer_inherits_when_valid
func.func @test_consumer_inherits_when_valid() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Producer with delay=50
    // CHECK: arith.constant {sched.delay = 50 : i32, sched.rate = 1 : i32} 1.0
    %prod = arith.constant {sched.delay = 50 : i32, sched.rate = 1 : i32} 1.0 : f32

    // Intermediate without schedule - will inherit from explicit consumer (delay=100)
    // which is >= operand delay (50), so no conflict
    // CHECK: arith.negf {{.*}} {sched.delay = 100 : i32, sched.rate = 2 : i32}
    %mid = arith.negf %prod : f32

    // Explicit consumer with higher delay
    %final = arith.addf %mid, %mid {sched.delay = 100 : i32, sched.rate = 2 : i32} : f32
  } {sched.dims = array<i64: 4>}
  return
}

// -----

// Test zero delay propagation when no operands have schedules (default case)
// CHECK-LABEL: func.func @test_zero_delay_no_operand_schedules
func.func @test_zero_delay_no_operand_schedules() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // All operations without explicit schedules - should get delay=0
    // CHECK: arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 1.0
    %a = arith.constant 1.0 : f32
    // CHECK: arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 2.0
    %b = arith.constant 2.0 : f32
    // CHECK: arith.addf {{.*}} {sched.delay = 0 : i32, sched.rate = 1 : i32}
    %c = arith.addf %a, %b : f32
  } {sched.dims = array<i64: 4>}
  return
}
