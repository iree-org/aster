// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// ============================================================================
// Three-stage pipeline: load / wait+compute / store
// Stage 0: async load (data + read_token)
// Stage 1: wait on token, compute on data
// Stage 2: make_register_range + store (write_token)
//
// Cross-stage values: token (0->1), data (0->1), computed (1->2)
// 3 iter_args in kernel.
//
// Prologue (2 sections):
//   Section 0: load(iter 0)
//   Section 1: load(iter 1), wait+compute(iter 0)
// Kernel: for i in [2, 6): all 3 stages active
// Epilogue (2 sections):
//   Section 1: wait+compute(iter 5), make_range+store(iter 4)
//   Section 2: make_range+store(iter 5)
// ============================================================================

// CHECK-LABEL: func.func @three_stage_load_compute_store

// Prologue section 0: load(iter 0)
// CHECK:       %[[P0_D:.*]], %[[P0_T:.*]] = amdgcn.load global_load_dword

// Prologue section 1: load(iter 1) + wait+compute(iter 0)
// Per-iteration mappings: iter 1's load results are separate from iter 0's.
// wait+compute uses iter 0's token/data (P0_T, P0_D), not iter 1's.
// CHECK:       %[[P1_D:.*]], %[[P1_T:.*]] = amdgcn.load global_load_dword
// CHECK:       amdgcn.wait deps %[[P0_T]]
// CHECK:       %[[P1_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[P0_D]]

// Kernel: 3 iter_args (token, data, computed)
// Token and data come from iter 1's load (P1_T, P1_D).
// Computed comes from iter 0's compute (P1_C).
// CHECK:       %[[KER:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A_T:.*]] = %[[P1_T]], %[[A_D:.*]] = %[[P1_D]], %[[A_C:.*]] = %[[P1_C]]) -> (!amdgcn.read_token<flat>, !amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[K_D:.*]], %[[K_T:.*]] = amdgcn.load global_load_dword
// CHECK:         amdgcn.wait deps %[[A_T]]
// CHECK:         %[[K_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[A_D]]
// CHECK:         amdgcn.make_register_range %[[A_C]]
// CHECK:         amdgcn.store global_store_dword
// CHECK:         scf.yield %[[K_T]], %[[K_D]], %[[K_C]] : !amdgcn.read_token<flat>, !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue section 1: wait+compute(iter 5) + make_range+store(iter 4)
// wait+compute uses kernel results #0, #1 (iter 5's stage 0 values).
// make_range+store uses kernel result #2 (iter 4's stage 1 value).
// CHECK:       amdgcn.wait deps %[[KER]]#0
// CHECK:       %[[E1_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1
// CHECK:       amdgcn.make_register_range %[[KER]]#2
// CHECK:       amdgcn.store global_store_dword

// Epilogue section 2: make_range+store(iter 5)
// Uses the value computed in section 1 for iter 5.
// CHECK:       amdgcn.make_register_range %[[E1_C]]
// CHECK:       amdgcn.store global_store_dword
// CHECK:       return

func.func @three_stage_load_compute_store(%addr: !amdgcn.vgpr<[? + 2]>) {
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %s_compute = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %data, %rtok = amdgcn.load global_load_dword dest %dest addr %addr {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
    %computed = amdgcn.test_inst outs %s_compute ins %data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %dr = amdgcn.make_register_range %computed {sched.stage = 2 : i32} : !amdgcn.vgpr
    %wtok = amdgcn.store global_store_dword data %dr addr %addr {sched.stage = 2 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  }
  return
}
