// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// ============================================================================
// Two-stage pipeline with DPS test_inst.
// Stage 0 produces a value, stage 1 consumes it.
// Cross-stage value: one !amdgcn.vgpr
//
// Original: for i in [0, 4): produce(i), consume(produce(i))
// Pipelined:
//   Prologue: produce(0)
//   Kernel:   for i in [1, 4): produce(i), consume(produce(i-1))
//   Epilogue: consume(produce(3))
// ============================================================================

// CHECK-LABEL: func.func @two_stage_dps

// Prologue: stage 0 at iteration 0
// CHECK:       %[[PRO:.*]] = amdgcn.test_inst outs %[[S0:.*]] : (!amdgcn.vgpr) -> !amdgcn.vgpr

// Kernel: cross-stage value as iter_arg, lb = 1
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[KER:.*]] = scf.for %[[KI:.*]] = %[[C1]] to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[PRO]]) -> (!amdgcn.vgpr)
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[ARG]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         scf.yield %[[K_V]] : !amdgcn.vgpr

// Epilogue: stage 1 using kernel result for iter 3
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       return

func.func @two_stage_dps() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Two-stage pipeline with amdgcn.load and token flow.
// Stage 0: async load (produces data + read_token)
// Stage 1: wait on token, compute on data
// Cross-stage values: !amdgcn.read_token<flat> (token) + !amdgcn.vgpr (data)
//
// This tests multi-result ops crossing stage boundaries: both results of
// amdgcn.load become separate iter_args in the kernel.
// ============================================================================

// CHECK-LABEL: func.func @two_stage_load_compute

// Prologue: load at iteration 0
// CHECK:       %[[PRO_D:.*]], %[[PRO_T:.*]] = amdgcn.load global_load_dword

// Kernel: token + data as iter_args, lb = 1
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[KER:.*]]:2 = scf.for %[[KI:.*]] = %[[C1]] to %{{.*}} step %{{.*}} iter_args(%[[A_T:.*]] = %[[PRO_T]], %[[A_D:.*]] = %[[PRO_D]]) -> (!amdgcn.read_token<flat>, !amdgcn.vgpr)
// CHECK:         %[[K_D:.*]], %[[K_T:.*]] = amdgcn.load global_load_dword
// CHECK:         amdgcn.wait deps %[[A_T]] : !amdgcn.read_token<flat>
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[A_D]]
// CHECK:         scf.yield %[[K_T]], %[[K_D]] : !amdgcn.read_token<flat>, !amdgcn.vgpr

// Epilogue: wait + compute using kernel results
// CHECK:       amdgcn.wait deps %[[KER]]#0 : !amdgcn.read_token<flat>
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1
// CHECK:       return

func.func @two_stage_load_compute(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %data, %tok = amdgcn.load global_load_dword dest %dest addr %addr {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
    %result = amdgcn.test_inst outs %s_compute ins %data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
