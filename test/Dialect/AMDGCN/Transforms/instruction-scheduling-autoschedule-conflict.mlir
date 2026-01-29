// RUN: aster-opt %s -amdgcn-instruction-scheduling-autoschedule -verify-diagnostics

// Test that autoschedule detects conflict when consumer delay < operand delay.
// This should emit an error because the intermediate operation cannot satisfy
// both constraints: operand requires delay >= 100, but consumer wants delay = 10.

func.func @test_autoschedule_conflict() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // Producer with high delay
    %producer = arith.constant {sched.delay = 100 : i32, sched.rate = 1 : i32} 1.0 : f32

    // Intermediate without explicit schedule - conflict!
    // Operand requires delay >= 100, but consumer below requires delay = 10
    // expected-warning@below {{autoschedule conflict: consumer 'arith.addf' requires delay=10, but operand constraints require delay>=100. Please add explicit sched.delay/sched.rate attributes to resolve this conflict.}}
    %intermediate = arith.negf %producer : f32

    // Consumer with low delay - causes conflict
    %final = arith.addf %intermediate, %intermediate {sched.delay = 10 : i32, sched.rate = 1 : i32} : f32
  } {sched.dims = array<i64: 4>}
  return
}
