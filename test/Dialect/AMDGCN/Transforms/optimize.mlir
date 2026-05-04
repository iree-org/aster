// RUN: aster-opt %s --amdgcn-optimize | FileCheck %s

// Global load with SGPR addr: both dynamic and constant offset from ptr_add
// are moved to the load. The ptr_add is eliminated entirely.
// CHECK-LABEL:   func.func @test_load_global_sgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C16:.*]] = arith.constant 16 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA]] addr %[[ARG0]] offset d(%[[ARG1]]) + c(%[[C16]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_global_sgpr_addr(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 16 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig1 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Global store with SGPR addr: same optimization as load.
// CHECK-LABEL:   func.func @test_store_global_sgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr) {
// CHECK-NEXT:      %[[C32:.*]] = arith.constant 32 : i32
// CHECK-NEXT:      %{{.*}} = amdgcn.global_store_dword data %[[ARG2]] addr %[[ARG0]] offset d(%[[ARG1]]) + c(%[[C32]]) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_store_global_sgpr_addr(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr) {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 32 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  %c0_i32_mig1 = arith.constant 0 : i32
  %token = amdgcn.global_store_dword data %arg2 addr %0 offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// DS load: only constant offset can be merged (dynamic offset cannot move).
// CHECK-LABEL:   func.func @test_load_ds_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C64:.*]] = arith.constant 64 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA]] addr %[[ARG0]] offset c(%[[C64]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_ds_const_offset(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 64 : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig1 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.ds_read_b32 dest %1 addr %0 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  return %dest_res : !amdgcn.vgpr
}

// Global load with VGPR addr: only constant offset merged, dynamic stays in ptr_add.
// CHECK-LABEL:   func.func @test_load_global_vgpr_addr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C8:.*]] = arith.constant 8 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[PTR:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA]] addr %[[PTR]] offset c(%[[C8]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_global_vgpr_addr(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 d_off = %arg1 c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig2 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Merge ptr_add const offset with existing load constant offset.
// CHECK-LABEL:   func.func @test_load_merge_const_offsets(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C24:.*]] = arith.constant 24 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[ALLOCA1]]) + c(%[[C24]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_merge_const_offsets(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %0 = amdgcn.ptr_add %arg0 c_off = 16 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c8) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// ptr_add with only constant offset, no dynamic - constant merged.
// CHECK-LABEL:   func.func @test_load_const_only(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C256:.*]] = arith.constant 256 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[ALLOCA1]]) + c(%[[C256]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_const_only(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 256 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig3 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c0_i32_mig3) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Constant offset exceeds limit (5000 > 4095). Constant is materialized as a
// VGPR mov and placed in the offset slot. ptr_add is eliminated.
// CHECK-LABEL:   func.func @test_load_large_const_no_fold(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C5000:.*]] = arith.constant 5000 : i32
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[MOV:.*]] = lsir.mov %[[ALLOCA1]], %[[C5000]] : !amdgcn.vgpr, i32
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[MOV]]) + c(%[[C0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_large_const_no_fold(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 5000 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig4 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c0_i32_mig4) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Merged constant would exceed limit (4096 + 8 = 4104). The ptr_add constant
// is materialized as a VGPR mov offset, existing const_offset (8) is preserved.
// CHECK-LABEL:   func.func @test_load_merge_exceeds_limit(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C4096:.*]] = arith.constant 4096 : i32
// CHECK-NEXT:      %[[C8:.*]] = arith.constant 8 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[MOV:.*]] = lsir.mov %[[ALLOCA1]], %[[C4096]] : !amdgcn.vgpr, i32
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[MOV]]) + c(%[[C8]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_merge_exceeds_limit(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %0 = amdgcn.ptr_add %arg0 c_off = 4096 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c8) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Boundary: 4095 is the max 13-bit signed positive offset, should fold.
// CHECK-LABEL:   func.func @test_load_boundary_4095(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C4095:.*]] = arith.constant 4095 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[ALLOCA1]]) + c(%[[C4095]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_boundary_4095(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 4095 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig5 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %dest_res : !amdgcn.vgpr
}

// Boundary: 4096 overflows 13-bit signed offset. Materialized as VGPR mov.
// CHECK-LABEL:   func.func @test_load_boundary_4096(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C4096:.*]] = arith.constant 4096 : i32
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[ALLOCA0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ALLOCA1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[MOV:.*]] = lsir.mov %[[ALLOCA1]], %[[C4096]] : !amdgcn.vgpr, i32
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.global_load_dword dest %[[ALLOCA0]] addr %[[ARG0]] offset d(%[[MOV]]) + c(%[[C0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_load_boundary_4096(%arg0: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %arg0 c_off = 4096 : !amdgcn.sgpr<[? + 2]>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig6 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.global_load_dword dest %1 addr %0 offset d(%voff) + c(%c0_i32_mig6) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
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
// CHECK-NEXT:      %[[C512:.*]] = arith.constant 512 : i32
// CHECK-NEXT:      %{{.*}} = amdgcn.ds_write_b64 data %[[ARG1]] addr %[[ARG0]] offset c(%[[C512]]) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_ds_write_addi_const(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %token = amdgcn.ds_write_b64 data %arg1 addr %1 offset c(%c0) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// DS read with lsir.addi constant offset: fold into ds_read offset.
// CHECK-LABEL:   func.func @test_ds_read_addi_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C1024:.*]] = arith.constant 1024 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA]] addr %[[ARG0]] offset c(%[[C1024]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_ds_read_addi_const(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c1024 = arith.constant 1024 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c1024 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.ds_read_b32 dest %2 addr %1 offset c(%c0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  return %dest_res : !amdgcn.vgpr
}

// -----

// DS write lsir.addi: merge constant with existing non-zero offset.
// CHECK-LABEL:   func.func @test_ds_write_addi_merge(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK-NEXT:      %[[C1544:.*]] = arith.constant 1544 : i32
// CHECK-NEXT:      %{{.*}} = amdgcn.ds_write_b64 data %[[ARG1]] addr %[[ARG0]] offset c(%[[C1544]]) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_ds_write_addi_merge(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c1536 = arith.constant 1536 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c1536 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c8 = arith.constant 8 : i32
  %token = amdgcn.ds_write_b64 data %arg1 addr %1 offset c(%c8) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// DS write lsir.addi: constant too large for 16-bit unsigned offset (>65535),
// should NOT fold.
// CHECK-LABEL:   func.func @test_ds_write_addi_too_large(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[C70000:.*]] = arith.constant 70000 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[ADDR:.*]] = lsir.addi i32 %[[ALLOCA]], %[[ARG0]], %[[C70000]] : !amdgcn.vgpr, !amdgcn.vgpr, i32
// CHECK-NEXT:      %{{.*}} = amdgcn.ds_write_b64 data %[[ARG1]] addr %[[ADDR]] offset c(%[[C0]]) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_ds_write_addi_too_large(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr<[? + 2]>) {
  %c70000 = arith.constant 70000 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c70000 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %token = amdgcn.ds_write_b64 data %arg1 addr %1 offset c(%c0) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// Buffer load with lsir.addi on voffset: fold constant into c() offset.
// Pattern: voffset = lsir.addi(base, 512) -> use base as voffset, 512 as c().
// CHECK-LABEL:   func.func @test_buffer_load_addi_const(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C512:.*]] = arith.constant 512 : i32
// CHECK-NEXT:      %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.buffer_load_dword dest %[[DEST]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF]]) + c(%[[C512]]) {offen} : outs(!amdgcn.vgpr) ins(<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_buffer_load_addi_const(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.buffer_load_dword dest %dest addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c0) {offen} : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load with lsir.addi: merge with existing non-zero c() offset.
// CHECK-LABEL:   func.func @test_buffer_load_addi_merge(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C520:.*]] = arith.constant 520 : i32
// CHECK-NEXT:      %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.buffer_load_dword dest %[[DEST]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF]]) + c(%[[C520]]) {offen} : outs(!amdgcn.vgpr) ins(<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_buffer_load_addi_merge(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c8 = arith.constant 8 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.buffer_load_dword dest %dest addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c8) {offen} : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load: constant exceeds 4095 (12-bit unsigned max), should NOT fold.
// CHECK-LABEL:   func.func @test_buffer_load_addi_too_large(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[C5000:.*]] = arith.constant 5000 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VOFF2:.*]] = lsir.addi i32 %[[ALLOCA]], %[[VOFF]], %[[C5000]] : !amdgcn.vgpr, !amdgcn.vgpr, i32
// CHECK-NEXT:      %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.buffer_load_dword dest %[[DEST]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF2]]) + c(%[[C0]]) {offen} : outs(!amdgcn.vgpr) ins(<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_buffer_load_addi_too_large(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c5000 = arith.constant 5000 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c5000 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.buffer_load_dword dest %dest addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c0) {offen} : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer load: boundary test - 4095 should fold (max 12-bit unsigned).
// CHECK-LABEL:   func.func @test_buffer_load_boundary_4095(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C4095:.*]] = arith.constant 4095 : i32
// CHECK-NEXT:      %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.buffer_load_dword dest %[[DEST]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF]]) + c(%[[C4095]]) {offen} : outs(!amdgcn.vgpr) ins(<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_buffer_load_boundary_4095(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr, %voff: !amdgcn.vgpr
) -> !amdgcn.vgpr {
  %c4095 = arith.constant 4095 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c4095 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %dest = amdgcn.alloca : !amdgcn.vgpr
  %result, %tok = amdgcn.buffer_load_dword dest %dest addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c0) {offen} : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Buffer store with lsir.addi on voffset: fold constant into c() offset.
// CHECK-LABEL:   func.func @test_buffer_store_addi_const(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[DATA:.*]]: !amdgcn.vgpr) {
// CHECK-NEXT:      %[[C512:.*]] = arith.constant 512 : i32
// CHECK-NEXT:      %{{.*}} = amdgcn.buffer_store_dword data %[[DATA]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF]]) + c(%[[C512]]) {offen} : ins(!amdgcn.vgpr, <[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_buffer_store_addi_const(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr,
    %voff: !amdgcn.vgpr, %data: !amdgcn.vgpr
) {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.buffer_store_dword data %data addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c0) {offen} : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----

// Buffer store with lsir.addi: merge with existing non-zero c() offset.
// CHECK-LABEL:   func.func @test_buffer_store_addi_merge(
// CHECK-SAME:      %[[RSRC:.*]]: !amdgcn.sgpr<[? + 4]>,
// CHECK-SAME:      %[[SOFF:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[DATA:.*]]: !amdgcn.vgpr) {
// CHECK-NEXT:      %[[C520:.*]] = arith.constant 520 : i32
// CHECK-NEXT:      %{{.*}} = amdgcn.buffer_store_dword data %[[DATA]] addr %[[RSRC]] offset u(%[[SOFF]]) + off_idx(%[[VOFF]]) + c(%[[C520]]) {offen} : ins(!amdgcn.vgpr, <[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_buffer_store_addi_merge(
    %rsrc: !amdgcn.sgpr<[? + 4]>, %soff: !amdgcn.sgpr,
    %voff: !amdgcn.vgpr, %data: !amdgcn.vgpr
) {
  %c512 = arith.constant 512 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %voff2 = lsir.addi i32 %0, %voff, %c512 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %c8 = arith.constant 8 : i32
  %tok = amdgcn.buffer_store_dword data %data addr %rsrc offset u(%soff) + off_idx(%voff2) + c(%c8) {offen} : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----

// DS read with lsir.addi: merge constant with existing non-zero offset.
// CHECK-LABEL:   func.func @test_ds_read_addi_merge(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK-NEXT:      %[[C1544:.*]] = arith.constant 1544 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %[[VAL:.*]], %[[TOK:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA]] addr %[[ARG0]] offset c(%[[C1544]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK-NEXT:      return %[[VAL]] : !amdgcn.vgpr
// CHECK-NEXT:    }
func.func @test_ds_read_addi_merge(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c1536 = arith.constant 1536 : i32
  %0 = lsir.alloca : !amdgcn.vgpr
  %1 = lsir.addi i32 %0, %arg0, %c1536 : !amdgcn.vgpr, !amdgcn.vgpr, i32
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %c8 = arith.constant 8 : i32
  %dest_res, %token = amdgcn.ds_read_b32 dest %2 addr %1 offset c(%c8) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  return %dest_res : !amdgcn.vgpr
}

// -----

// Global store with SGPR addr, const-only ptr_add: constant offset is moved
// to the store's offset operand (no dynamic offset on ptr_add).
// CHECK-LABEL:   func.func @test_store_global_const_only(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK-NEXT:      %[[C256:.*]] = arith.constant 256 : i32
// CHECK-NEXT:      %[[ALLOCA:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK-NEXT:      %{{.*}} = amdgcn.global_store_dword data %[[ARG1]] addr %[[ARG0]] offset d(%[[ALLOCA]]) + c(%[[C256]]) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @test_store_global_const_only(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr) {
  %0 = amdgcn.ptr_add %arg0 c_off = 256 : !amdgcn.sgpr<[? + 2]>
  %voff = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  %token = amdgcn.global_store_dword data %arg1 addr %0 offset d(%voff) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
  return
}
