// RUN: aster-opt '--pass-pipeline=builtin.module(aster-scf-pipeline{gcd-unroll=true})' %s | FileCheck %s

// CHECK-LABEL: func.func @gap_0_3_static_ub

// Prologue: 3 iterations
// CHECK:       amdgcn.test_inst outs %[[T0:.*]] :
// CHECK:       amdgcn.test_inst outs %[[T0]] :
// CHECK:       amdgcn.test_inst outs %[[T0]] :

// Main kernel: unrolled by 3
// CHECK:       scf.for
// CHECK:         amdgcn.test_inst outs %[[T0]]
// CHECK:         amdgcn.test_inst outs %[[T1:.*]] ins
// CHECK:         amdgcn.test_inst outs %[[T0]]
// CHECK:         amdgcn.test_inst outs %[[T1]] ins
// CHECK:         amdgcn.test_inst outs %[[T0]]
// CHECK:         amdgcn.test_inst outs %[[T1]] ins
// CHECK:         scf.yield

// Unroll cleanup loop: 2 remainder iterations
// CHECK:       scf.for
// CHECK:         amdgcn.test_inst outs %[[T0]]
// CHECK:         amdgcn.test_inst outs %[[T1]] ins
// CHECK:         scf.yield

// Epilogue: three consume ops
// CHECK:       amdgcn.test_inst outs %[[T1]] ins
// CHECK:       amdgcn.test_inst outs %[[T1]] ins
// CHECK:       amdgcn.test_inst outs %[[T1]] ins
// CHECK:       return

func.func @gap_0_3_static_ub() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  scf.for %i = %c0 to %c11 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 3 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// CHECK-LABEL: func.func @gap_0_3_dynamic_ub
// CHECK-SAME:    (%[[UB:.*]]: index)

// Prologue: 3 iterations
// CHECK:       amdgcn.test_inst outs %[[S0:.*]] :
// CHECK:       amdgcn.test_inst outs %[[S0]] :
// CHECK:       amdgcn.test_inst outs %[[S0]] :

// Main kernel: unrolled by 3
// CHECK:       scf.for
// CHECK:         amdgcn.test_inst outs %[[S0]]
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins
// CHECK:         amdgcn.test_inst outs %[[S0]]
// CHECK:         amdgcn.test_inst outs %[[S1]] ins
// CHECK:         amdgcn.test_inst outs %[[S0]]
// CHECK:         amdgcn.test_inst outs %[[S1]] ins
// CHECK:         scf.yield

// Unroll cleanup loop: 2 remainder iterations
// CHECK:       scf.for {{.*}} to %[[UB]] step
// CHECK:         amdgcn.test_inst outs %[[S0]]
// CHECK:         amdgcn.test_inst outs %[[S1]] ins
// CHECK:         scf.yield

// Epilogue: three consume ops
// CHECK:       affine.apply {{.*}}(%[[UB]])
// CHECK:       amdgcn.test_inst outs %[[S1]] ins
// CHECK:       affine.apply {{.*}}(%[[UB]])
// CHECK:       amdgcn.test_inst outs %[[S1]] ins
// CHECK:       affine.apply {{.*}}(%[[UB]])
// CHECK:       amdgcn.test_inst outs %[[S1]] ins
// CHECK:       return

func.func @gap_0_3_dynamic_ub(%ub: index) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 3 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
