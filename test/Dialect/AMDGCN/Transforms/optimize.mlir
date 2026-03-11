// RUN: aster-opt %s --amdgcn-optimize | FileCheck %s

// Global load with SGPR addr: both dynamic and constant offset from ptr_add
// are moved to the load. The ptr_add is simplified to just pass-through.
// CHECK-LABEL:   func.func @test_load_global_sgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset d(%[[ARG1]]) + c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_global_sgpr_addr(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 16 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Global store with SGPR addr: same optimization as load.
// CHECK-LABEL:   func.func @test_store_global_sgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG2]] addr %[[ARG0]] offset d(%[[ARG1]]) + c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_sgpr_addr(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr) {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 32 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  %token = amdgcn.store global_store_dword data %arg2 addr %0 : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// DS load: only constant offset can be merged (dynamic offset cannot move).
// CHECK-LABEL:   func.func @test_load_ds_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_ds_const_offset(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 64 : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load ds_read_b32 dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  return %dest_res : !amdgcn.vgpr
}

// Global load with VGPR addr: only constant offset merged, dynamic stays in ptr_add.
// CHECK-LABEL:   func.func @test_load_global_vgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[PTR_ADD_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_global_vgpr_addr(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Merge ptr_add const offset with existing load constant offset.
// CHECK-LABEL:   func.func @test_load_merge_const_offsets(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 24 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_merge_const_offsets(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %0 = amdgcn.ptr_add %arg0 c_off = 16 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 offset c(%c8) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// ptr_add with only constant offset, no dynamic - constant merged.
// CHECK-LABEL:   func.func @test_load_const_only(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 256 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_const_only(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 256 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// No update: constant offset exceeds limit (4096). ptr_add remains unchanged.
// CHECK-LABEL:   func.func @test_load_large_const_no_fold(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5000 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MOV_0:.*]] = lsir.mov %[[ALLOCA_1]], %[[CONSTANT_0]] : !amdgcn.vgpr, i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset d(%[[MOV_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_large_const_no_fold(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 5000 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// No update: merged constant would exceed limit (4096 + 8 = 4104).
// CHECK-LABEL:   func.func @test_load_merge_exceeds_limit(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4096 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MOV_0:.*]] = lsir.mov %[[ALLOCA_1]], %[[CONSTANT_0]] : !amdgcn.vgpr, i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset d(%[[MOV_0]]) + c(%[[CONSTANT_1]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_merge_exceeds_limit(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %0 = amdgcn.ptr_add %arg0 c_off = 4096 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 offset c(%c8) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Boundary: 4095 is the max 13-bit signed positive offset, should fold.
// CHECK-LABEL:   func.func @test_load_boundary_4095(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[C4095:.*]] = arith.constant 4095 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%[[C4095]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_boundary_4095(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 4095 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Boundary: 4096 overflows 13-bit signed offset.
// CHECK-LABEL:   func.func @test_load_boundary_4096(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4096 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MOV_0:.*]] = lsir.mov %[[ALLOCA_1]], %[[CONSTANT_0]] : !amdgcn.vgpr, i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[ARG0]] offset d(%[[MOV_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_boundary_4096(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 4096 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// -----

// DS write with lsir.addi constant offset: fold constant into ds_write offset.
// This is the pattern from kittens 32x32 GEMM LDS writes where intra-tile
// row offsets (512, 1024, 1536) are added via lsir.addi but could be absorbed
// into the ds_write_b64 offset field (16-bit unsigned, max 65535).
// CHECK-LABEL:   func.func @test_ds_write_addi_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[C512:.*]] = arith.constant 512 : i32
// CHECK:           %[[TOKEN:.*]] = amdgcn.store ds_write_b64 data %[[ARG1]] addr %[[ARG0]] offset c(%[[C512]])
// CHECK:           return
// CHECK:         }
func.func @test_ds_write_addi_const(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %token = amdgcn.store ds_write_b64 data %arg1 addr %1 offset c(%c0) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// DS read with lsir.addi constant offset: fold into ds_read offset.
// CHECK-LABEL:   func.func @test_ds_read_addi_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[C1024:.*]] = arith.constant 1024 : i32
// CHECK:           %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], %[[TOKEN:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA]] addr %[[ARG0]] offset c(%[[C1024]])
// CHECK:           return %[[VAL]]
// CHECK:         }
func.func @test_ds_read_addi_const(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c1024 = arith.constant 1024 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c1024 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.load ds_read_b32 dest %2 addr %1 offset c(%c0) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  return %dest_res : !amdgcn.vgpr
}

// -----

// DS write lsir.addi: merge constant with existing non-zero offset.
// CHECK-LABEL:   func.func @test_ds_write_addi_merge(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[C1544:.*]] = arith.constant 1544 : i32
// CHECK:           %[[TOKEN:.*]] = amdgcn.store ds_write_b64 data %[[ARG1]] addr %[[ARG0]] offset c(%[[C1544]])
// CHECK:           return
// CHECK:         }
func.func @test_ds_write_addi_merge(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c1536 = arith.constant 1536 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c1536 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c8 = arith.constant 8 : i32
  %token = amdgcn.store ds_write_b64 data %arg1 addr %1 offset c(%c8) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// DS write lsir.addi: constant too large for 16-bit unsigned offset (>65535),
// should NOT fold.
// CHECK-LABEL:   func.func @test_ds_write_addi_too_large(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[C70000:.*]] = arith.constant 70000 : i32
// CHECK:           %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ADDR:.*]] = lsir.addi i32 %[[ALLOCA]], %[[ARG0]], %[[C70000]]
// CHECK:           %[[TOKEN:.*]] = amdgcn.store ds_write_b64 data %[[ARG1]] addr %[[ADDR]]
// CHECK:           return
// CHECK:         }
func.func @test_ds_write_addi_too_large(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c70000 = arith.constant 70000 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c70000 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %token = amdgcn.store ds_write_b64 data %arg1 addr %1 offset c(%c0) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// Buffer load with lsir.addi on voffset: fold constant into c() offset.
// Pattern: voffset = lsir.addi(base, 512) -> use base as voffset, 512 as c().
// CHECK-LABEL:   func.func @test_buffer_load_addi_const(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[C512:.*]] = arith.constant 512 : i32
// CHECK:           %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], %[[TOK:.*]] = amdgcn.load buffer_load_dword dest %[[DEST]] addr %[[RSRC]]
// CHECK-SAME:        offset u(%[[SOFF]]) + d(%[[VOFF]]) + c(%[[C512]])
// CHECK:           return %[[VAL]]
// CHECK:         }
func.func @test_buffer_load_addi_const(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.load buffer_load_dword dest %dest addr %rsrc
      offset u(%soff) + d(%voff2) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load with lsir.addi: merge with existing non-zero c() offset.
// CHECK-LABEL:   func.func @test_buffer_load_addi_merge(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[C520:.*]] = arith.constant 520 : i32
// CHECK:           %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], %[[TOK:.*]] = amdgcn.load buffer_load_dword dest %[[DEST]] addr %[[RSRC]]
// CHECK-SAME:        offset u(%[[SOFF]]) + d(%[[VOFF]]) + c(%[[C520]])
// CHECK:           return %[[VAL]]
// CHECK:         }
func.func @test_buffer_load_addi_merge(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c8 = arith.constant 8 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.load buffer_load_dword dest %dest addr %rsrc
      offset u(%soff) + d(%voff2) + c(%c8)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load: constant exceeds 4095 (12-bit unsigned max), should NOT fold.
// CHECK-LABEL:   func.func @test_buffer_load_addi_too_large(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[C5000:.*]] = arith.constant 5000 : i32
// CHECK:           %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VOFF2:.*]] = lsir.addi i32 %[[ALLOCA]], %[[VOFF]], %[[C5000]]
// CHECK:           %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], %[[TOK:.*]] = amdgcn.load buffer_load_dword dest %[[DEST]] addr %[[RSRC]]
// CHECK-SAME:        offset u(%[[SOFF]]) + d(%[[VOFF2]]) + c(
// CHECK:           return %[[VAL]]
// CHECK:         }
func.func @test_buffer_load_addi_too_large(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c5000 = arith.constant 5000 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c5000 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.load buffer_load_dword dest %dest addr %rsrc
      offset u(%soff) + d(%voff2) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load: boundary test - 4095 should fold (max 12-bit unsigned).
// CHECK-LABEL:   func.func @test_buffer_load_boundary_4095(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[C4095:.*]] = arith.constant 4095 : i32
// CHECK:           %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], %[[TOK:.*]] = amdgcn.load buffer_load_dword dest %[[DEST]] addr %[[RSRC]]
// CHECK-SAME:        offset u(%[[SOFF]]) + d(%[[VOFF]]) + c(%[[C4095]])
// CHECK:           return %[[VAL]]
// CHECK:         }
func.func @test_buffer_load_boundary_4095(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c4095 = arith.constant 4095 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c4095 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.load buffer_load_dword dest %dest addr %rsrc
      offset u(%soff) + d(%voff2) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}
