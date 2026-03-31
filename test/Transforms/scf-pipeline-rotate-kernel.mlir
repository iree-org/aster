// RUN: aster-opt %s --aster-scf-pipeline="enable-rotate-stage=true rotate-stage=1" | FileCheck %s

!vgpr = !amdgcn.vgpr

// CHECK-LABEL: func.func @rotate_2stage
// CHECK-NOT:   sched.stage
// Prologue: a single load seeds the data shift register.
// CHECK:       %[[INIT_DATA:.*]] = amdgcn.test_inst outs %{{.*}} {load} : (!amdgcn.vgpr) -> !amdgcn.vgpr
//
// Kernel: 2 iter_args = (load shift, acc carry). Compute leads, load follows.
// CHECK:       %[[KER:.*]]:2 = scf.for {{.*}} iter_args(%[[DATA_PREV:.*]] = %[[INIT_DATA]], %[[ACC:.*]] = %{{.*}}) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[NEW_ACC:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[ACC]], %[[DATA_PREV]] {compute}
// CHECK:         %[[NEW_DATA:.*]] = amdgcn.test_inst outs %{{.*}} {load}
// CHECK:         scf.yield %[[NEW_DATA]], %[[NEW_ACC]] : !amdgcn.vgpr, !amdgcn.vgpr
//
// Epilogue: drain compute consuming the final kernel state.
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1, %[[KER]]#0 {compute}

func.func @rotate_2stage(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr
  %s1 = amdgcn.alloca : !vgpr
  %s_out = amdgcn.alloca : !vgpr
  %init = amdgcn.test_inst outs %s0 : (!vgpr) -> !vgpr

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr {
    %data = amdgcn.test_inst outs %s1
        {load, sched.stage = 0 : i32} : (!vgpr) -> !vgpr

    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %data
        {compute, sched.stage = 1 : i32}
        : (!vgpr, !vgpr, !vgpr) -> !vgpr

    scf.yield %new_acc : !vgpr
  }
  return
}
