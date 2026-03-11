// RUN: aster-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @merge_waits(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.read_token<flat>, %[[ARG1:.*]]: !amdgcn.read_token<shared>, %[[ARG2:.*]]: !amdgcn.write_token<flat>) {
// CHECK:           amdgcn.wait vm_cnt 0 lgkm_cnt 1 deps %[[ARG0]], %[[ARG1]], %[[ARG2]] : !amdgcn.read_token<flat>, !amdgcn.read_token<shared>, !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @merge_waits(
    %rt1: !amdgcn.read_token<flat>,
    %rt2: !amdgcn.read_token<shared>,
    %wt1: !amdgcn.write_token<flat>) {
  amdgcn.wait deps %rt1 : !amdgcn.read_token<flat>
  amdgcn.wait deps %rt1, %rt2 : !amdgcn.read_token<flat>, !amdgcn.read_token<shared>
  amdgcn.wait deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  amdgcn.wait vm_cnt 0 lgkm_cnt 1 deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  amdgcn.wait vm_cnt 2
  return
}

// CHECK-LABEL:   func.func @remove_duplicate_waits(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.read_token<flat>) {
// CHECK:           amdgcn.wait deps %[[ARG0]] : !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @remove_duplicate_waits(%rt1: !amdgcn.read_token<flat>) {
  amdgcn.wait deps %rt1, %rt1, %rt1 : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @erase_noop_wait() {
// CHECK:           return
// CHECK:         }
func.func @erase_noop_wait() {
  amdgcn.wait
  return
}

// CHECK-LABEL:   func.func @lds_buffer_folding(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> (!amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer) {
// CHECK-DAG:       %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds %[[ARG0]]
// CHECK-DAG:       %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 64
// CHECK-DAG:       %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 32
// CHECK:           return %[[ALLOC_LDS_0]], %[[ALLOC_LDS_1]], %[[ALLOC_LDS_2]] : !amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer
// CHECK:         }
func.func @lds_buffer_folding(%arg0: index) -> (!amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer) {
  %c64 = arith.constant 64 : index
  %0 = amdgcn.alloc_lds %arg0
  %1 = amdgcn.alloc_lds %c64
  %2 = amdgcn.alloc_lds 32
  return %0, %1, %2 : !amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer
}

// CHECK-LABEL:   func.func @ptr_add_no_offset_fold(
// CHECK-SAME:      %[[PTR:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           return %[[PTR]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_no_offset_fold(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr : !amdgcn.vgpr<[? + 2]>
  return %result : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @fold_no_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @fold_no_offset(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr : !amdgcn.vgpr<[? + 2]>
  return %result : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @fold_no_offset_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           return %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @fold_no_offset_sgpr(%ptr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr : !amdgcn.sgpr<[? + 2]>
  return %result : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @no_fold_with_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_fold_with_const_offset(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %result : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @no_fold_with_dynamic_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_fold_with_dynamic_offset(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr d_off = %off : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %result : !amdgcn.vgpr<[? + 2]>
}

// Merge two ptr_adds that both have only const offsets.
// ptr_add(ptr_add(base, c_off=16), c_off=32) → ptr_add(base, c_off=48)
// CHECK-LABEL:   func.func @merge_const_offsets(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 48 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_const_offsets(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner c_off = 32 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Merge two ptr_adds on an SGPR pointer with only const offsets.
// CHECK-LABEL:   func.func @merge_const_offsets_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 48 : !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @merge_const_offsets_sgpr(%ptr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.sgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner c_off = 32 : !amdgcn.sgpr<[? + 2]>
  return %outer : !amdgcn.sgpr<[? + 2]>
}

// Outer has a dynamic offset, inner has only const offset.
// ptr_add(ptr_add(base, c_off=8), d_off=%off, c_off=16) → ptr_add(base, d_off=%off, c_off=24)
// CHECK-LABEL:   func.func @merge_outer_dynamic_inner_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] c_off = 24 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_outer_dynamic_inner_const(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 8 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner d_off = %off c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Inner has a dynamic offset, outer has only const offset.
// ptr_add(ptr_add(base, d_off=%off, c_off=8), c_off=16) → ptr_add(base, d_off=%off, c_off=24)
// CHECK-LABEL:   func.func @merge_inner_dynamic_outer_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] c_off = 24 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_inner_dynamic_outer_const(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr d_off = %off c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  %outer = amdgcn.ptr_add %inner c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Inner has dynamic offset with no const, outer has only const offset.
// ptr_add(ptr_add(base, d_off=%off), c_off=16) → ptr_add(base, d_off=%off, c_off=16)
// CHECK-LABEL:   func.func @merge_inner_dynamic_only_outer_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_inner_dynamic_only_outer_const(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr d_off = %off : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  %outer = amdgcn.ptr_add %inner c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Both ptr_adds have matching `inbounds` flags — should merge.
// CHECK-LABEL:   func.func @merge_matching_flags_inbounds(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add inbounds %[[ARG0]] c_off = 24 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_matching_flags_inbounds(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add inbounds %ptr c_off = 8 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add inbounds %inner c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// SGPR base with dynamic offset from outer — result should become VGPR.
// CHECK-LABEL:   func.func @merge_sgpr_base_outer_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] c_off = 24 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @merge_sgpr_base_outer_dynamic(%ptr: !amdgcn.sgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 8 : !amdgcn.sgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner d_off = %off c_off = 16 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Base is not a PtrAddOp — should not merge.
// CHECK-LABEL:   func.func @no_merge_base_not_ptr_add(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_base_not_ptr_add(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %result = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %result : !amdgcn.vgpr<[? + 2]>
}

// Flags don't match — should not merge.
// CHECK-LABEL:   func.func @no_merge_flags_mismatch(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add inbounds %[[ARG0]] c_off = 8 : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[PTR_ADD_1:.*]] = amdgcn.ptr_add %[[PTR_ADD_0]] c_off = 16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_flags_mismatch(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add inbounds %ptr c_off = 8 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Outer has a uniform offset — should not merge.
// CHECK-LABEL:   func.func @no_merge_outer_uniform_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 8 : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[PTR_ADD_1:.*]] = amdgcn.ptr_add %[[PTR_ADD_0]] u_off = %[[ARG1]] c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr
// CHECK:           return %[[PTR_ADD_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_outer_uniform_offset(%ptr: !amdgcn.vgpr<[? + 2]>, %uoff: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 8 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner u_off = %uoff c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Inner has a uniform offset — should not merge.
// CHECK-LABEL:   func.func @no_merge_inner_uniform_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] u_off = %[[ARG1]] c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr
// CHECK:           %[[PTR_ADD_1:.*]] = amdgcn.ptr_add %[[PTR_ADD_0]] c_off = 16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_inner_uniform_offset(%ptr: !amdgcn.vgpr<[? + 2]>, %uoff: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr u_off = %uoff c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr
  %outer = amdgcn.ptr_add %inner c_off = 16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Both have dynamic offsets — should not merge.
// CHECK-LABEL:   func.func @no_merge_both_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] d_off = %[[ARG1]] c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           %[[PTR_ADD_1:.*]] = amdgcn.ptr_add %[[PTR_ADD_0]] d_off = %[[ARG2]] c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
// CHECK:           return %[[PTR_ADD_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_both_dynamic(%ptr: !amdgcn.vgpr<[? + 2]>, %off1: !amdgcn.vgpr, %off2: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr d_off = %off1 c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  %outer = amdgcn.ptr_add %inner d_off = %off2 c_off = 16 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %outer : !amdgcn.vgpr<[? + 2]>
}

// Const offset is negative
// CHECK-LABEL:   func.func @no_merge_negative_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[PTR_ADD_0:.*]] = amdgcn.ptr_add %[[ARG0]] c_off = 16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[PTR_ADD_1:.*]] = amdgcn.ptr_add %[[PTR_ADD_0]] c_off = -16 : !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[PTR_ADD_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @no_merge_negative_const_offset(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %inner = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.vgpr<[? + 2]>
  %outer = amdgcn.ptr_add %inner c_off = -16 : !amdgcn.vgpr<[? + 2]>
  return %outer : !amdgcn.vgpr<[? + 2]>
}
