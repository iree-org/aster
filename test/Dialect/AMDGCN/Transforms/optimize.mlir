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
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 5000 : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[PTR_ADD_0]] : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<flat>
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
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 4096 : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_0]] addr %[[PTR_ADD_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_merge_exceeds_limit(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %0 = amdgcn.ptr_add %arg0 c_off = 4096 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %1 addr %0 offset c(%c8) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}
