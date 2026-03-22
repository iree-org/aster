// RUN: aster-opt %s --sched-expand-resources | FileCheck %s

// CHECK-LABEL: func.func @test_expand_minimal
// The loop_resource should be gone; the allocate value should be available
// before the loop and forward values replace the resource results.
// CHECK-NOT: sched.loop_resource
// CHECK: arith.constant 42
// CHECK: scf.for
func.func @test_expand_minimal() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %token = sched.loop_resource -> i32 {
      %buf = arith.constant 42 : i32
      sched.yield %buf : i32
    } forward {
    ^bb0(%buf: i32):
      sched.yield %buf : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @test_expand_with_deallocate
// CHECK-NOT: sched.loop_resource
// CHECK: arith.constant 7
// CHECK: scf.for
// Deallocate inline after the loop:
// CHECK: arith.constant 0
func.func @test_expand_with_deallocate() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %token = sched.loop_resource -> i32 {
      %buf = arith.constant 7 : i32
      sched.yield %buf : i32
    } deallocate {
    ^bb0(%buf: i32):
      %zero = arith.constant 0 : i32
      sched.yield
    } forward {
    ^bb0(%buf: i32):
      sched.yield %buf : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @test_expand_with_fence
// CHECK-NOT: sched.loop_resource
// Fence should appear at end of loop body.
func.func @test_expand_with_fence() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %token = sched.loop_resource -> i32 {
      %buf = arith.constant 99 : i32
      sched.yield %buf : i32
    } forward {
    ^bb0(%buf: i32):
      sched.yield %buf : i32
    } fence {
    ^bb0(%buf: i32):
      sched.yield
    }
  }
  return
}
