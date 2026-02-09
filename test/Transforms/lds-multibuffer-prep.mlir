// RUN: aster-opt --test-lds-multibuffer-prep %s | FileCheck %s

// Two-stage LDS: alloc_lds at stage 0, dealloc_lds at stage 1.
// N = 1 - 0 + 1 = 2 buffers needed for double-buffering.

// CHECK-LABEL: func.func @two_stage_lds
//
// Two hoisted allocs + offsets before loop
// CHECK-DAG:   %[[LDS0:.*]] = amdgcn.alloc_lds 256
// CHECK-DAG:   %[[OFF0:.*]] = amdgcn.get_lds_offset %[[LDS0]] : i32
// CHECK-DAG:   %[[LDS1:.*]] = amdgcn.alloc_lds 256
// CHECK-DAG:   %[[OFF1:.*]] = amdgcn.get_lds_offset %[[LDS1]] : i32
//
// Loop with two i32 iter_args for rotating offsets
// CHECK:       scf.for {{.*}} iter_args(%[[CUR:.*]] = %[[OFF0]], %[[PREV:.*]] = %[[OFF1]]) -> (i32, i32)
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// CHECK:         lsir.to_reg %[[CUR]]
// CHECK:         amdgcn.store ds_write_b32
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// CHECK:         scf.yield %[[PREV]], %[[CUR]]
//
// Deallocs after loop
// CHECK:       amdgcn.dealloc_lds %[[LDS0]]
// CHECK:       amdgcn.dealloc_lds %[[LDS1]]
// CHECK:       return

func.func @two_stage_lds(%data_in: !amdgcn.vgpr, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c4 step %c1 {
    // Stage 0: alloc LDS, get offset, write
    %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.store ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 1: wait, read, compute, dealloc
    amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %read_data, %rtok = amdgcn.load ds_read_b32 dest %dest addr %lds_addr {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %read_data {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
  }
  return
}

// Three-stage with two independent LDS groups:
//   Group A: alloc stage 0, dealloc stage 1 -> N=2
//   Group B: alloc stage 1, dealloc stage 2 -> N=2
// Total: 4 hoisted buffers, 4 iter_args.

// CHECK-LABEL: func.func @two_groups_three_stage
//
// Four hoisted allocs (2 per group)
// CHECK-DAG:   %[[A0:.*]] = amdgcn.alloc_lds 128
// CHECK-DAG:   %[[OA0:.*]] = amdgcn.get_lds_offset %[[A0]] : i32
// CHECK-DAG:   %[[A1:.*]] = amdgcn.alloc_lds 128
// CHECK-DAG:   %[[OA1:.*]] = amdgcn.get_lds_offset %[[A1]] : i32
// CHECK-DAG:   %[[B0:.*]] = amdgcn.alloc_lds 128
// CHECK-DAG:   %[[OB0:.*]] = amdgcn.get_lds_offset %[[B0]] : i32
// CHECK-DAG:   %[[B1:.*]] = amdgcn.alloc_lds 128
// CHECK-DAG:   %[[OB1:.*]] = amdgcn.get_lds_offset %[[B1]] : i32
//
// Loop with 4 i32 iter_args: group A offsets, then group B offsets
// CHECK:       scf.for {{.*}} iter_args(%[[CA:.*]] = %[[OA0]], %[[PA:.*]] = %[[OA1]], %[[CB:.*]] = %[[OB0]], %[[PB:.*]] = %[[OB1]]) -> (i32, i32, i32, i32)
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// Group A uses its current iter_arg
// CHECK:         lsir.to_reg %[[CA]]
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// Group B uses its current iter_arg
// CHECK:         lsir.to_reg %[[CB]]
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// Yield rotates both groups
// CHECK:         scf.yield %[[PA]], %[[CA]], %[[PB]], %[[CB]]
//
// Four deallocs after loop, tied to hoisted handles
// CHECK:       amdgcn.dealloc_lds %[[A0]]
// CHECK:       amdgcn.dealloc_lds %[[A1]]
// CHECK:       amdgcn.dealloc_lds %[[B0]]
// CHECK:       amdgcn.dealloc_lds %[[B1]]
// CHECK:       return

func.func @two_groups_three_stage(%data_in: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  scf.for %i = %c0 to %c6 step %c1 {
    // Group A: stage 0 -> stage 1
    %lds_a = amdgcn.alloc_lds 128 {sched.stage = 0 : i32}
    %off_a = amdgcn.get_lds_offset %lds_a {sched.stage = 0 : i32} : i32
    %addr_a = lsir.to_reg %off_a {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok_a = amdgcn.store ds_write_b32 data %data_in addr %addr_a offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 1: read from A, dealloc A, alloc B, write to B
    amdgcn.wait deps %wtok_a {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest_a = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %from_a, %rtok_a = amdgcn.load ds_read_b32 dest %dest_a addr %addr_a {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok_a {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    amdgcn.dealloc_lds %lds_a {sched.stage = 1 : i32}

    // Group B: stage 1 -> stage 2
    %lds_b = amdgcn.alloc_lds 128 {sched.stage = 1 : i32}
    %off_b = amdgcn.get_lds_offset %lds_b {sched.stage = 1 : i32} : i32
    %addr_b = lsir.to_reg %off_b {sched.stage = 1 : i32} : i32 -> !amdgcn.vgpr
    %wtok_b = amdgcn.store ds_write_b32 data %from_a addr %addr_b offset c(%c0_i32) {sched.stage = 1 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 2: read from B, compute, dealloc B
    amdgcn.wait deps %wtok_b {sched.stage = 2 : i32} : !amdgcn.write_token<shared>
    %dest_b = amdgcn.alloca {sched.stage = 2 : i32} : !amdgcn.vgpr
    %from_b, %rtok_b = amdgcn.load ds_read_b32 dest %dest_b addr %addr_b {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok_b {sched.stage = 2 : i32} : !amdgcn.read_token<shared>
    %result = amdgcn.test_inst outs %s_out ins %from_b {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds_b {sched.stage = 2 : i32}
  }
  return
}

// Two-stage LDS with pre-existing iter_args on the scf.for.

// CHECK-LABEL: func.func @existing_iter_args
//
// Two hoisted allocs + offsets before loop
// CHECK-DAG:   %[[LDS0:.*]] = amdgcn.alloc_lds 256
// CHECK-DAG:   %[[OFF0:.*]] = amdgcn.get_lds_offset %[[LDS0]] : i32
// CHECK-DAG:   %[[LDS1:.*]] = amdgcn.alloc_lds 256
// CHECK-DAG:   %[[OFF1:.*]] = amdgcn.get_lds_offset %[[LDS1]] : i32
//
// Loop: existing iter_arg first, then LDS offset iter_args
// CHECK:       %[[LOOP:.*]]:3 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}, %[[CUR:.*]] = %[[OFF0]], %[[PREV:.*]] = %[[OFF1]]) -> (!amdgcn.vgpr, i32, i32)
// CHECK-NOT:     amdgcn.alloc_lds
// CHECK-NOT:     amdgcn.get_lds_offset
// CHECK-NOT:     amdgcn.dealloc_lds
// CHECK:         lsir.to_reg %[[CUR]]
// CHECK:         amdgcn.store ds_write_b32
// CHECK:         %[[NEW_ACC:.*]] = amdgcn.test_inst
// CHECK:         scf.yield %[[NEW_ACC]], %[[PREV]], %[[CUR]]
//
// Deallocs after loop
// CHECK:       amdgcn.dealloc_lds %[[LDS0]]
// CHECK:       amdgcn.dealloc_lds %[[LDS1]]
// CHECK:       return

func.func @existing_iter_args(%data_in: !amdgcn.vgpr, %init_acc: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %s_out = amdgcn.alloca : !amdgcn.vgpr
  %result = scf.for %i = %c0 to %c4 step %c1
      iter_args(%acc = %init_acc) -> (!amdgcn.vgpr) {
    // Stage 0: alloc LDS, get offset, write
    %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
    %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
    %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %wtok = amdgcn.store ds_write_b32 data %data_in addr %lds_addr offset c(%c0_i32) {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Stage 1: wait, read, accumulate, dealloc
    amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
    %dest = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.vgpr
    %read_data, %rtok = amdgcn.load ds_read_b32 dest %dest addr %lds_addr {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>
    %new_acc = amdgcn.test_inst outs %s_out ins %acc {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
    scf.yield %new_acc : !amdgcn.vgpr
  }
  return %result : !amdgcn.vgpr
}

// Negative test: loop without sched.stage annotations is left untouched.

// CHECK-LABEL: func.func @no_stage_annotations
// CHECK:       scf.for
// CHECK-NOT:     iter_args
// CHECK:         %[[LDS:.*]] = amdgcn.alloc_lds 64
// CHECK:         amdgcn.get_lds_offset %[[LDS]]
// CHECK:         amdgcn.dealloc_lds %[[LDS]]
// CHECK:       return

func.func @no_stage_annotations() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %lds = amdgcn.alloc_lds 64
    %off = amdgcn.get_lds_offset %lds : i32
    amdgcn.dealloc_lds %lds
  }
  return
}
