// RUN: aster-opt %s --sched-pipeline-loop="target-stage=0" | FileCheck %s

// A simple 2-stage pipeline.
// Stage 0 is the target: prologue executes stage 0's body before the loop.
// Epilogue executes stage 1's body after the loop.
// The loop body is left unchanged.
// CHECK-LABEL: func.func @test_pipeline_two_stages
// Prologue: stage 0 body inlined before the loop.
// CHECK: %[[ADD:.+]] = arith.addi %arg0, %arg0
// CHECK: scf.for
// CHECK:   sched.unit
// CHECK-SAME: stage 0
// CHECK:   sched.unit
// CHECK-SAME: stage 1
// Epilogue: stage 1 body inlined after the loop.
// CHECK: arith.muli %[[ADD]], %[[ADD]]
func.func @test_pipeline_two_stages(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = sched.unit(%arg0 : i32) -> (i32) stage 0 {
    ^bb0(%x: i32):
      %v = arith.addi %x, %x : i32
      sched.yield %v : i32
    }
    %1 = sched.unit(%0 : i32) -> (i32) stage 1 {
    ^bb0(%y: i32):
      %v = arith.muli %y, %y : i32
      sched.yield %v : i32
    }
  }
  return
}
