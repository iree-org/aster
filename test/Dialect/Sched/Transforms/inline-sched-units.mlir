// RUN: aster-opt %s --sched-inline-units | FileCheck %s

// CHECK-LABEL: func.func @test_inline_simple
// CHECK-NOT: sched.unit
// CHECK: %[[C:.*]] = arith.addi
// CHECK: return %[[C]]
func.func @test_inline_simple(%arg0: i32, %arg1: i32) -> i32 {
  %0 = sched.unit(%arg0, %arg1 : i32, i32) -> (i32) stage 0 {
  ^bb0(%x: i32, %y: i32):
    %c = arith.addi %x, %y : i32
    sched.yield %c : i32
  }
  return %0 : i32
}

// CHECK-LABEL: func.func @test_inline_multiple_results
// CHECK-NOT: sched.unit
// CHECK: %[[A:.*]] = arith.addi
// CHECK: %[[B:.*]] = arith.muli
// CHECK: return %[[A]], %[[B]]
func.func @test_inline_multiple_results(%a: i32, %b: i32) -> (i32, i32) {
  %0, %1 = sched.unit(%a, %b : i32, i32) -> (i32, i32) stage 1 {
  ^bb0(%x: i32, %y: i32):
    %xa = arith.addi %x, %y : i32
    %xb = arith.muli %x, %y : i32
    sched.yield %xa, %xb : i32, i32
  }
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @test_inline_no_io
// CHECK-NOT: sched.unit
// CHECK: return
func.func @test_inline_no_io() {
  sched.unit stage 0 {
    sched.yield
  }
  return
}
