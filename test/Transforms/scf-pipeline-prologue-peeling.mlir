// RUN: aster-opt '--pass-pipeline=builtin.module(aster-scf-pipeline{prologue-peeling=2 lcm-unroll=true unroll-factor-multiplier=2})' %s | FileCheck %s

// CHECK-LABEL: func.func @two_stage_dynamic
//
// CHECK:       amdgcn.test_inst outs %[[U_S0:.*]] :
//
// First peeled unrolled-body
// CHECK:       amdgcn.test_inst outs %[[U_S0]] :
// CHECK:       amdgcn.test_inst outs %[[U_S1:.*]] ins
// CHECK:       amdgcn.test_inst outs %[[U_S0]] :
// CHECK:       amdgcn.test_inst outs %[[U_S1]] ins
//
// Second peeled unrolled-body
// CHECK:       amdgcn.test_inst outs %[[U_S0]] :
// CHECK:       amdgcn.test_inst outs %[[U_S1]] ins
// CHECK:       amdgcn.test_inst outs %[[U_S0]] :
// CHECK:       amdgcn.test_inst outs %[[U_S1]] ins
//
// CHECK:       scf.for
// CHECK:         amdgcn.test_inst outs %[[U_S0]]
// CHECK:         amdgcn.test_inst outs %[[U_S1]] ins
// CHECK:         amdgcn.test_inst outs %[[U_S0]]
// CHECK:         amdgcn.test_inst outs %[[U_S1]] ins
// CHECK:         scf.yield
// CHECK:       return

func.func @two_stage_dynamic(%ub: index) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
