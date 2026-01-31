// RUN: aster-opt --aster-op-scheduling="test-only=true" --allow-unregistered-dialect %s | FileCheck %s

// Test that loop without sched.dims is silently skipped (no error, no transformation).

// CHECK-LABEL: func.func @test_no_sched_dims
// CHECK: scf.for
// CHECK: arith.constant 1.0
func.func @test_no_sched_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // No sched.dims attribute - silently skipped, loop preserved
  scf.for %i = %c0 to %c4 step %c1 {
    %val = arith.constant 1.0 : f32
  }
  return
}
