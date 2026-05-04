// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// ============================================================================
// Multiple independent cross-stage values.
// Stage 0 produces two independent values, stage 1 consumes both.
// Cross-stage values: %a and %b (both !amdgcn.vgpr)
// 2 iter_args in kernel.
// ============================================================================

// CHECK-LABEL: func.func @two_producers_one_consumer

// Prologue: both stage-0 ops at iter 0
// CHECK:       %[[P_A:.*]] = amdgcn.test_inst outs %[[S0:.*]] :
// CHECK:       %[[P_B:.*]] = amdgcn.test_inst outs %[[S1:.*]] :

// Kernel: 2 iter_args, lb = 1
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[KER:.*]]:2 = scf.for %{{.*}} = %[[C1]] to %{{.*}} step %{{.*}} iter_args(%[[AA:.*]] = %[[P_A]], %[[AB:.*]] = %[[P_B]]) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[K_A:.*]] = amdgcn.test_inst outs %[[S0]] :
// CHECK:         %[[K_B:.*]] = amdgcn.test_inst outs %[[S1]] :
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[AA]], %[[AB]]
// CHECK:         scf.yield %[[K_A]], %[[K_B]] : !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue: consume kernel results for iter 3
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#0, %[[KER]]#1
// CHECK:       return

func.func @two_producers_one_consumer() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = amdgcn.test_inst outs %s1 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %c = amdgcn.test_inst outs %s2 ins %a, %b {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Load with token: both data and token cross the stage boundary.
// Tests that multi-result ops (amdgcn.load -> data, token) generate
// separate iter_args for each cross-stage result.
// ============================================================================

// CHECK-LABEL: func.func @load_data_and_token_cross_stage

// Prologue: load at iteration 0
// CHECK:       %[[PRO_D:.*]], %[[PRO_T:.*]] = amdgcn.load global_load_dword

// Kernel: token + data as iter_args, lb = 1
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[KER:.*]]:2 = scf.for %{{.*}} = %[[C1]] to %{{.*}} step %{{.*}} iter_args(%[[A_T:.*]] = %[[PRO_T]], %[[A_D:.*]] = %[[PRO_D]]) -> (!amdgcn.read_token<flat>, !amdgcn.vgpr)
// CHECK:         %[[K_D:.*]], %[[K_T:.*]] = amdgcn.load global_load_dword
// CHECK:         amdgcn.wait deps %[[A_T]] : !amdgcn.read_token<flat>
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[A_D]]
// CHECK:         scf.yield %[[K_T]], %[[K_D]] : !amdgcn.read_token<flat>, !amdgcn.vgpr

// Epilogue
// CHECK:       amdgcn.wait deps %[[KER]]#0
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1
// CHECK:       return

func.func @load_data_and_token_cross_stage(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %data, %tok = amdgcn.load global_load_dword dest %dest addr %addr {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
    %out = amdgcn.test_inst outs %s_out ins %data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
