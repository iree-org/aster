// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @two_stage_with_iter_args

// Prologue: stage 0 at iteration 0 -- to_reg + produce using init0
// CHECK-DAG:   %[[INIT0:.*]] = arith.constant 10
// CHECK-DAG:   %[[INIT1:.*]] = arith.constant 20
// CHECK:       %[[P_ADDR:.*]] = lsir.to_reg %[[INIT0]] : i32 -> !amdgcn.vgpr
// CHECK:       %[[PRO:.*]] = amdgcn.test_inst outs %[[S0:.*]] ins %[[P_ADDR]] :

// Kernel: 1 cross-stage + 2 existing iter_args
// After prologue yield sim: a=init1, b=init0
// CHECK:       %[[KER:.*]]:3 = scf.for
// CHECK-SAME:    iter_args(%[[CSV:.*]] = %[[PRO]], %[[KA:.*]] = %[[INIT1]], %[[KB:.*]] = %[[INIT0]])

// Kernel stage 0: to_reg + produce using iter_arg ka
// CHECK:         %[[K_ADDR:.*]] = lsir.to_reg %[[KA]] : i32 -> !amdgcn.vgpr
// CHECK:         %[[KV:.*]] = amdgcn.test_inst outs %[[S0]] ins %[[K_ADDR]] :

// Kernel stage 1: consume using cross-stage from prev iter
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[CSV]] :

// Yield: cross-stage + swapped existing iter_args
// CHECK:         scf.yield %[[KV]], %[[KB]], %[[KA]] : !amdgcn.vgpr, i32, i32

// Epilogue: stage 1 using final cross-stage
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 :
// CHECK:       return

func.func @two_stage_with_iter_args() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %init0 = arith.constant 10 : i32
  %init1 = arith.constant 20 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%a = %init0, %b = %init1) -> (i32, i32) {
    %addr = lsir.to_reg %a {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %v = amdgcn.test_inst outs %s0 ins %addr {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %b, %a : i32, i32
  }
  return
}

// CHECK-LABEL: func.func @three_stage_with_iter_args
// CHECK-DAG:   %[[I0:.*]] = arith.constant 10
// CHECK-DAG:   %[[I1:.*]] = arith.constant 20

// Prologue section 0: stage 0 at iter 0
// CHECK:       lsir.to_reg %[[I0]] : i32 -> !amdgcn.vgpr
// CHECK:       %[[P0_V:.*]] = amdgcn.test_inst

// Prologue section 1: stage 0 at iter 1, stage 1 at iter 0
// CHECK:       lsir.to_reg %[[I1]] : i32 -> !amdgcn.vgpr
// CHECK:       %[[P1_V:.*]] = amdgcn.test_inst
// CHECK:       %[[P1_W:.*]] = amdgcn.test_inst{{.*}} ins %[[P0_V]] :

// Kernel: 2 cross-stage + 2 existing iter_args
// After 2 yield sims: (init0,init1) -> (init1,init0) -> (init0,init1)
// CHECK:       %[[KER:.*]]:4 = scf.for
// CHECK-SAME:    iter_args(%[[CA:.*]] = %[[P1_V]], %[[CB:.*]] = %[[P1_W]], %[[KA:.*]] = %[[I0]], %[[KB:.*]] = %[[I1]])

// Kernel stage 0: to_reg + produce using %ka
// CHECK:         lsir.to_reg %[[KA]]
// CHECK:         %[[KV:.*]] = amdgcn.test_inst
// Kernel stage 1: consume cross-stage %ca, produce
// CHECK:         %[[KW:.*]] = amdgcn.test_inst{{.*}} ins %[[CA]] :
// Kernel stage 2: consume cross-stage %cb
// CHECK:         amdgcn.test_inst{{.*}} ins %[[CB]] :
// Yield: cross-stage + swapped existing
// CHECK:         scf.yield %[[KV]], %[[KW]], %[[KB]], %[[KA]] : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32

// Epilogue section 1: stages 1+2
// CHECK:       amdgcn.test_inst{{.*}} ins %[[KER]]#0
// CHECK:       amdgcn.test_inst{{.*}} ins %[[KER]]#1

// Epilogue section 2: stage 2 only
// CHECK:       amdgcn.test_inst
// CHECK:       return

func.func @three_stage_with_iter_args() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %init0 = arith.constant 10 : i32
  %init1 = arith.constant 20 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %r:2 = scf.for %i = %c0 to %c6 step %c1
      iter_args(%a = %init0, %b = %init1) -> (i32, i32) {
    %addr = lsir.to_reg %a {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %v = amdgcn.test_inst outs %s0 ins %addr {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %x = amdgcn.test_inst outs %s2 ins %w {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %b, %a : i32, i32
  }
  return
}

// CHECK-LABEL: func.func @no_cross_stage_with_used_result

// Prologue: stage 0 only (unused op)
// CHECK:       amdgcn.test_inst

// Kernel: 0 cross-stage + 1 existing iter_arg, init = %init (unchanged)
// CHECK:       %[[KER:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[KACC:.*]] = %[[INIT:.*]]) -> (!amdgcn.vgpr)
// CHECK:         amdgcn.test_inst
// CHECK:         %[[K_NEW:.*]] = amdgcn.test_inst{{.*}} ins %[[KACC]]
// CHECK:         scf.yield %[[K_NEW]] : !amdgcn.vgpr

// Epilogue: stage 1 using kernel result
// CHECK:       %[[EPI:.*]] = amdgcn.test_inst{{.*}} ins %[[KER]]

// Original loop result replaced with epilogue result
// CHECK:       amdgcn.make_register_range %[[EPI]]
// CHECK:       return

func.func @no_cross_stage_with_used_result() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %init = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (!amdgcn.vgpr) {
    %unused = amdgcn.test_inst outs %s0 ins %s1 {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %new_acc = amdgcn.test_inst outs %s1 ins %acc {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_acc : !amdgcn.vgpr
  }
  %dr = amdgcn.make_register_range %r : !amdgcn.vgpr
  return
}
