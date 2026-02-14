// RUN: aster-opt --aster-scf-pipeline --amdgcn-lds-alloc --canonicalize --cse %s | FileCheck %s

// Two-stage LDS pipeline: alloc_lds hoisted before loop, i32 offset iter_args
// rotate each iteration (double-buffering). No alloc_lds inside loop body.

// CHECK-LABEL: func.func @lds_write_read_two_stage
// CHECK-SAME:  attributes {gpu.shared_memory_size = 512 : i32}

// Two hoisted allocs with concrete LDS offsets
// CHECK:       %[[LDS0:.*]] = amdgcn.alloc_lds 256 offset 0
// CHECK:       %[[OFF0:.*]] = amdgcn.get_lds_offset %[[LDS0]] : i32
// CHECK:       %[[LDS1:.*]] = amdgcn.alloc_lds 256 offset 256
// CHECK:       %[[OFF1:.*]] = amdgcn.get_lds_offset %[[LDS1]] : i32

// Prologue: stage 0
// CHECK:       %[[P_ADDR:.*]] = lsir.to_reg %[[OFF0]] : i32 -> !amdgcn.vgpr
// CHECK:       %[[P_WTOK:.*]] = amdgcn.store ds_write_b32

// Kernel: 4 iter_args (2 cross-stage + 2 offsets)
// CHECK:       scf.for {{.*}} iter_args(%[[A_WTOK:.*]] = %[[P_WTOK]], %[[A_ADDR:.*]] = %[[P_ADDR]], %[[CUR:.*]] = %[[OFF1]], %[[PREV:.*]] = %[[OFF0]]) -> (!amdgcn.write_token<shared>, !amdgcn.vgpr, i32, i32)

// Stage 0: write
// CHECK:         %[[K_ADDR:.*]] = lsir.to_reg %[[CUR]] : i32 -> !amdgcn.vgpr
// CHECK:         %[[K_WTOK:.*]] = amdgcn.store ds_write_b32

// Stage 1: read from previous iteration's buffer
// CHECK:         amdgcn.wait deps %[[A_WTOK]]
// CHECK:         amdgcn.load ds_read_b32 dest %{{.*}} addr %[[A_ADDR]]
// CHECK:         amdgcn.test_inst

// Offsets rotate: (cur, prev) -> (prev, cur)
// CHECK:         scf.yield %[[K_WTOK]], %[[K_ADDR]], %[[PREV]], %[[CUR]]

// Epilogue: stage 1 drains the last iteration's LDS
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       amdgcn.test_inst

// Deallocs after everything
// CHECK:       amdgcn.dealloc_lds %[[LDS0]]
// CHECK:       amdgcn.dealloc_lds %[[LDS1]]
// CHECK:       return

func.func @lds_write_read_two_stage(%data_in: !amdgcn.vgpr, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c4 step %c1 {
    // Stage 0: alloc LDS, write data into it
    %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.store ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 1: wait, read from LDS, compute, dealloc
    amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %read_data, %rtok = amdgcn.load ds_read_b32 dest %dest addr %lds_addr {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %read_data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
  }
  return
}

// Three-stage double-buffer cascade: 2 LDS groups (A stages 0-1, B stages
// 1-2), 4 hoisted allocs. No alloc_lds/dealloc_lds inside loop body.

// CHECK-LABEL: func.func @double_lds_three_stage
// CHECK-SAME:  attributes {gpu.shared_memory_size = 512 : i32}

// Four hoisted allocs with concrete offsets (2 per group)
// CHECK:       %[[A0:.*]] = amdgcn.alloc_lds 128 offset 0
// CHECK:       %[[OA0:.*]] = amdgcn.get_lds_offset %[[A0]] : i32
// CHECK:       %[[A1:.*]] = amdgcn.alloc_lds 128 offset 128
// CHECK:       %[[OA1:.*]] = amdgcn.get_lds_offset %[[A1]] : i32
// CHECK:       %[[B0:.*]] = amdgcn.alloc_lds 128 offset 256
// CHECK:       %[[OB0:.*]] = amdgcn.get_lds_offset %[[B0]] : i32
// CHECK:       %[[B1:.*]] = amdgcn.alloc_lds 128 offset 384
// CHECK:       %[[OB1:.*]] = amdgcn.get_lds_offset %[[B1]] : i32

// Prologue section 0
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32

// Prologue section 1: stage 0 + stage 1
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32

// Kernel: 8 iter_args (4 cross-stage + 4 offsets)
// CHECK:       scf.for {{.*}} -> (!amdgcn.write_token<shared>, !amdgcn.vgpr, !amdgcn.write_token<shared>, !amdgcn.vgpr, i32, i32, i32, i32)

// Kernel body: all 3 stages active
// Stage 0: write
// CHECK:         lsir.to_reg
// CHECK:         amdgcn.store ds_write_b32
// Stage 1: read prev group A, write group B
// CHECK:         amdgcn.wait
// CHECK:         amdgcn.load ds_read_b32
// CHECK:         lsir.to_reg
// CHECK:         amdgcn.store ds_write_b32
// Stage 2: read prev group B, compute
// CHECK:         amdgcn.wait
// CHECK:         amdgcn.load ds_read_b32
// CHECK:         amdgcn.test_inst
// No alloc_lds inside loop body
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK:         scf.yield

// Epilogue drains remaining stages
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       amdgcn.test_inst
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       amdgcn.test_inst

// All 4 deallocs after the loop
// CHECK:       amdgcn.dealloc_lds %[[A0]]
// CHECK:       amdgcn.dealloc_lds %[[A1]]
// CHECK:       amdgcn.dealloc_lds %[[B0]]
// CHECK:       amdgcn.dealloc_lds %[[B1]]
// CHECK:       return

func.func @double_lds_three_stage(%gaddr: !amdgcn.vgpr<[? + 2]>, %data_in: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c6 step %c1 {
    // Stage 0: alloc LDS_A, write into it
    %lds_a = amdgcn.alloc_lds 128 {sched.stage = 0 : i32}
    %off_a = amdgcn.get_lds_offset %lds_a {sched.stage = 0 : i32} : i32
    %addr_a = lsir.to_reg %off_a {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok_a = amdgcn.store ds_write_b32 data %data_in addr %addr_a offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 1: read from LDS_A, alloc LDS_B, write to LDS_B
    amdgcn.wait deps %wtok_a {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest_a = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %from_a, %rtok_a = amdgcn.load ds_read_b32 dest %dest_a addr %addr_a {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok_a {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    amdgcn.dealloc_lds %lds_a {sched.stage = 1 : i32}
    %lds_b = amdgcn.alloc_lds 128 {sched.stage = 1 : i32}
    %off_b = amdgcn.get_lds_offset %lds_b {sched.stage = 1 : i32} : i32
    %addr_b = lsir.to_reg %off_b {sched.stage = 1 : i32} : i32 -> !amdgcn.vgpr
    %wtok_b = amdgcn.store ds_write_b32 data %from_a addr %addr_b offset c(%c0_i32) {sched.stage = 1 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 2: read from LDS_B, compute, dealloc
    amdgcn.wait deps %wtok_b {sched.stage = 2 : i32} : !amdgcn.write_token<shared>
    %dest_b = amdgcn.alloca {sched.stage = 2 : i32} : !amdgcn.vgpr
    %from_b, %rtok_b = amdgcn.load ds_read_b32 dest %dest_b addr %addr_b {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok_b {sched.stage = 2 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %from_b {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds_b {sched.stage = 2 : i32}
  }
  return
}

// Global load -> LDS write -> LDS read -> compute (3-stage GEMM prefetch
// pattern). LDS allocated stage 1, deallocated stage 2 -> 2 buffers hoisted.

// CHECK-LABEL: func.func @global_to_lds_to_compute
// CHECK-SAME:  attributes {gpu.shared_memory_size = 512 : i32}

// Two hoisted LDS allocs with concrete offsets
// CHECK:       %[[LDS0:.*]] = amdgcn.alloc_lds 256 offset 0
// CHECK:       %[[OFF0:.*]] = amdgcn.get_lds_offset %[[LDS0]] : i32
// CHECK:       %[[LDS1:.*]] = amdgcn.alloc_lds 256 offset 256
// CHECK:       %[[OFF1:.*]] = amdgcn.get_lds_offset %[[LDS1]] : i32

// Prologue section 0
// CHECK:       amdgcn.load global_load_dword

// Prologue section 1
// CHECK:       amdgcn.load global_load_dword
// CHECK:       amdgcn.wait
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32

// Kernel: 6 iter_args (4 cross-stage + 2 offsets)
// CHECK:       scf.for {{.*}} -> (!amdgcn.read_token<flat>, !amdgcn.vgpr, !amdgcn.write_token<shared>, !amdgcn.vgpr, i32, i32)

// Stage 0
// CHECK:         amdgcn.load global_load_dword

// Stage 1: wait on prev global load, write to LDS
// CHECK:         amdgcn.wait
// CHECK:         lsir.to_reg
// CHECK:         amdgcn.store ds_write_b32

// Stage 2: read from prev LDS, compute
// CHECK:         amdgcn.wait
// CHECK:         amdgcn.load ds_read_b32
// CHECK:         amdgcn.test_inst
// CHECK:         scf.yield

// Epilogue
// CHECK:       amdgcn.wait
// CHECK:       lsir.to_reg
// CHECK:       amdgcn.store ds_write_b32
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       amdgcn.test_inst
// CHECK:       amdgcn.wait
// CHECK:       amdgcn.load ds_read_b32
// CHECK:       amdgcn.test_inst

// Deallocs
// CHECK:       amdgcn.dealloc_lds %[[LDS0]]
// CHECK:       amdgcn.dealloc_lds %[[LDS1]]
// CHECK:       return

func.func @global_to_lds_to_compute(%gaddr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32
  %g_dest = amdgcn.alloca : !amdgcn.vgpr
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c6 step %c1 {
    // Stage 0: async global load
    %gdata, %gtok = amdgcn.load global_load_dword dest %g_dest addr %gaddr {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>

    // Stage 1: wait for global load, write to LDS
    amdgcn.wait deps %gtok {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
    %lds = amdgcn.alloc_lds 256 {sched.stage = 1 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 1 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 1 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.store ds_write_b32 data %gdata addr %lds_addr offset c(%c0_i32) {sched.stage = 1 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 2: read from LDS, compute, dealloc
    amdgcn.wait deps %wtok {sched.stage = 2 : i32} : !amdgcn.write_token<shared>
    %r_dest = amdgcn.alloca {sched.stage = 2 : i32} : !amdgcn.vgpr
    %from_lds, %rtok = amdgcn.load ds_read_b32 dest %r_dest addr %lds_addr {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 2 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %from_lds {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 2 : i32}
  }
  return
}
