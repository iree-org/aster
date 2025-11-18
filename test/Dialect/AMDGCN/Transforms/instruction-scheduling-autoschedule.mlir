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
