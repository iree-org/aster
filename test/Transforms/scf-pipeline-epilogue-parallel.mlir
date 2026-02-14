// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @epilogue_parallel_lanes

// Prologue section 0: load(iter 0)
// Per-iteration mappings: iter 0 gets its own namespace.
// CHECK:       %[[P0_D:.*]], %[[P0_T:.*]] = amdgcn.load global_load_dword

// Prologue section 1: load(iter 1) + wait + compute(iter 0)
// Per-iteration mappings: iter 1's load results are separate from iter 0's.
// wait+compute uses iter 0's token/data (P0_T, P0_D), not iter 1's.
// CHECK:       %[[P1_D:.*]], %[[P1_T:.*]] = amdgcn.load global_load_dword
// CHECK:       amdgcn.wait deps %[[P0_T]]
// CHECK:       %[[P1_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[P0_D]]

// Kernel: 3 iter_args
// Token and data from iter 1's load, computed from iter 0's compute.
// CHECK:       %[[KER:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A_T:.*]] = %[[P1_T]], %[[A_D:.*]] = %[[P1_D]], %[[A_C:.*]] = %[[P1_C]]) -> (!amdgcn.read_token<flat>, !amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[K_D:.*]], %[[K_T:.*]] = amdgcn.load global_load_dword
// CHECK:         amdgcn.wait deps %[[A_T]]
// CHECK:         %[[K_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[A_D]]
// CHECK:         amdgcn.make_register_range %[[A_C]]
// CHECK:         amdgcn.store global_store_dword
// CHECK:         scf.yield %[[K_T]], %[[K_D]], %[[K_C]] : !amdgcn.read_token<flat>, !amdgcn.vgpr, !amdgcn.vgpr

// -- Epilogue section 1 --
// Stage 1 ops for iter 5:
// CHECK:       amdgcn.wait deps %[[KER]]#0
// CHECK:       %[[E1_C:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1

// Stage 2 ops for iter 4: MUST use %[[KER]]#2, NOT %[[E1_C]]
// CHECK:       amdgcn.make_register_range %[[KER]]#2
// CHECK:       amdgcn.store global_store_dword

// -- Epilogue section 2 --
// Stage 2 ops for iter 5: uses the freshly computed value from section 1
// CHECK:       amdgcn.make_register_range %[[E1_C]]
// CHECK:       amdgcn.store global_store_dword
// CHECK:       return

func.func @epilogue_parallel_lanes(%addr: !amdgcn.vgpr<[? + 2]>) {
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
