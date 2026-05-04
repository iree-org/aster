// RUN: aster-opt %s --aster-apply-sched=scheds=sched --allow-unregistered-dialect | FileCheck %s

#sched = #aster_utils.generic_scheduler<#amdgcn.value_scheduler, #aster_utils.sched_stage_labeler, #aster_utils.stage_topo_sort_sched>

// CHECK-LABEL:   func.func @amdgcn_load_wait_store(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG1]] addr %[[ARG0]] {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.wait deps %[[LOAD_0]] {sched.stage = 3 : i32} : !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[VAL_0]] addr %[[ARG0]] {sched.stage = 4 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.wait deps %[[STORE_0]], %[[LOAD_0]] {sched.stage = 5 : i32} : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @amdgcn_load_wait_store(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: !amdgcn.vgpr) attributes {sched = #sched} {
  %dest_res, %token = amdgcn.load global_load_dword dest %arg1 addr %arg0 {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token {sched.stage = 3 : i32} : !amdgcn.read_token<flat>
  %0 = amdgcn.store global_store_dword data %dest_res addr %arg0 {sched.stage = 4 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  amdgcn.wait deps %0, %token {sched.stage = 5 : i32} : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @amdgcn_multiple_loads(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG1]] addr %[[ARG0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.wait deps %[[LOAD_0]] {sched.stage = 0 : i32} : !amdgcn.read_token<flat>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ARG1]] addr %[[ARG0]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ARG1]] addr %[[ARG0]] {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.wait deps %[[LOAD_1]] {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @amdgcn_multiple_loads(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: !amdgcn.vgpr) attributes {sched = #sched} {
  %dest_res, %token = amdgcn.load global_load_dword dest %arg1 addr %arg0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %dest_res_0, %token_1 = amdgcn.load global_load_dword dest %arg1 addr %arg0 {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %dest_res_2, %token_3 = amdgcn.load global_load_dword dest %arg1 addr %arg0 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token {sched.stage = 0 : i32} : !amdgcn.read_token<flat>
  amdgcn.wait deps %token_3 {sched.stage = 1 : i32} : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @amdgcn_mixed_memory_spaces(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG3:.*]]: !amdgcn.sgpr, %[[ARG4:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ARG3]] addr %[[ARG0]] {sched.stage = 1 : i32} : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load ds_read_b32 dest %[[ARG4]] addr %[[ARG1]] {sched.stage = 4 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           amdgcn.wait deps %[[LOAD_0]] {sched.stage = 2 : i32} : !amdgcn.read_token<constant>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ARG4]] addr %[[ARG2]] {sched.stage = 5 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @amdgcn_mixed_memory_spaces(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr<[? + 2]>, %arg3: !amdgcn.sgpr, %arg4: !amdgcn.vgpr) attributes {sched = #sched} {
  %dest_res, %token = amdgcn.load s_load_dword dest %arg3 addr %arg0 {sched.stage = 1 : i32} : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
  %dest_res_0, %token_1 = amdgcn.load ds_read_b32 dest %arg4 addr %arg1 {sched.stage = 4 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_2, %token_3 = amdgcn.load global_load_dword dest %arg4 addr %arg2 {sched.stage = 5 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token {sched.stage = 2 : i32} : !amdgcn.read_token<constant>
  return
}

// CHECK-LABEL:   func.func @amdgcn_barrier(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr, %[[ARG3:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[SCC_ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.scc<0>
// CHECK:           %[[VAL_0:.*]] = amdgcn.s_add_u32 outs(%[[ARG2]], %[[SCC_ALLOCA_0]]) ins(%[[ARG0]], %[[ARG1]]) {sched.stage = 1 : i32} : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
// CHECK:           amdgcn.s_barrier {sched.stage = 0 : i32}
// CHECK:           %[[VAL_1:.*]] = amdgcn.s_mul_i32 outs(%[[ARG3]]) ins(%[[VAL_0]], %[[ARG0]]) {sched.stage = 2 : i32} : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
// CHECK:           return %[[VAL_1]] : !amdgcn.sgpr
// CHECK:         }
func.func @amdgcn_barrier(%arg0: !amdgcn.sgpr, %arg1: !amdgcn.sgpr, %arg2: !amdgcn.sgpr, %arg3: !amdgcn.sgpr) -> !amdgcn.sgpr attributes {sched = #sched} {
  %_scc_dst_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %0 = amdgcn.s_add_u32 outs(%arg2, %_scc_dst_add_u32) ins(%arg0, %arg1) {sched.stage = 1 : i32} : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.s_barrier {sched.stage = 0 : i32}
  %1 = amdgcn.s_mul_i32 outs(%arg3) ins(%0, %arg0) {sched.stage = 2 : i32} : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %1 : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @amdgcn_vop2_salu_barrier(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.sgpr, %[[ARG3:.*]]: !amdgcn.sgpr, %[[ARG4:.*]]: !amdgcn.vgpr, %[[ARG5:.*]]: !amdgcn.sgpr) -> (!amdgcn.vgpr, !amdgcn.sgpr) {
// CHECK:           %[[VAL_0:.*]] = amdgcn.v_add_u32 outs(%[[ARG4]]) ins(%[[ARG0]], %[[ARG1]]) {sched.stage = 2 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[SCC_ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.scc<0>
// CHECK:           %[[VAL_1:.*]] = amdgcn.s_add_u32 outs(%[[ARG5]], %[[SCC_ALLOCA_0]]) ins(%[[ARG2]], %[[ARG3]]) {sched.stage = 1 : i32} : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
// CHECK:           amdgcn.s_barrier {sched.stage = 0 : i32}
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.sgpr
// CHECK:         }
func.func @amdgcn_vop2_salu_barrier(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.sgpr, %arg3: !amdgcn.sgpr, %arg4: !amdgcn.vgpr, %arg5: !amdgcn.sgpr) -> (!amdgcn.vgpr, !amdgcn.sgpr) attributes {sched = #sched} {
  %vdst0_res = amdgcn.v_add_u32 outs(%arg4) ins(%arg0, %arg1) {sched.stage = 2 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  %_scc_dst_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %0 = amdgcn.s_add_u32 outs(%arg5, %_scc_dst_add_u32) ins(%arg2, %arg3) {sched.stage = 1 : i32} : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.s_barrier {sched.stage = 0 : i32}
  return %vdst0_res, %0 : !amdgcn.vgpr, !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @promote_vmem_forward() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.wait lgkm_cnt 0 {sched.stage = 0 : i32}
// CHECK:           amdgcn.s_barrier {sched.stage = 0 : i32}
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @promote_vmem_forward() attributes {sched = #sched} {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %dest_res_0, %token_1 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait lgkm_cnt 0 {sched.stage = 0 : i32}
  amdgcn.s_barrier {sched.stage = 0 : i32}
  %dest_res_2, %token_3 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL:   func.func @promote_vmem_backward() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.wait lgkm_cnt 0 {sched.stage = 1 : i32}
// CHECK:           amdgcn.s_barrier {sched.stage = 1 : i32}
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           return
// CHECK:         }
func.func @promote_vmem_backward() attributes {sched = #sched} {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait lgkm_cnt 0 {sched.stage = 1 : i32}
  amdgcn.s_barrier {sched.stage = 1 : i32}
  %dest_res_0, %token_1 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_2, %token_3 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @cant_promote_vmem_forward() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.wait vm_cnt 0 lgkm_cnt 0 {sched.stage = 0 : i32}
// CHECK:           amdgcn.s_barrier {sched.stage = 0 : i32}
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           return
// CHECK:         }
func.func @cant_promote_vmem_forward() attributes {sched = #sched} {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %dest_res_0, %token_1 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait vm_cnt 0 lgkm_cnt 0 {sched.stage = 0 : i32}
  amdgcn.s_barrier {sched.stage = 0 : i32}
  %dest_res_2, %token_3 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL:   func.func @cant_promote_vmem_backward() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.wait vm_cnt 0 lgkm_cnt 0 {sched.stage = 1 : i32}
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.s_barrier {sched.stage = 1 : i32}
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           return
// CHECK:         }
func.func @cant_promote_vmem_backward() attributes {sched = #sched} {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait vm_cnt 0 lgkm_cnt 0 {sched.stage = 1 : i32}
  amdgcn.s_barrier {sched.stage = 1 : i32}
  %dest_res_0, %token_1 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 2 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_2, %token_3 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @cant_promote_across_unknown_op() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 1 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           "test.barrier"() : () -> ()
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           return
// CHECK:         }
func.func @cant_promote_across_unknown_op() attributes {sched = #sched} {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 1 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  "test.barrier"() : () -> ()
  %dest_res_0, %token_1 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_2, %token_3 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @promote_pure_op_forward() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.wait lgkm_cnt 0 {sched.stage = 0 : i32}
// CHECK:           amdgcn.s_barrier {sched.stage = 0 : i32}
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[ALLOCA_1]] {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {sched.stage = 4 : i32} 0 : i32
// CHECK:           return
// CHECK:         }
func.func @promote_pure_op_forward() attributes {sched = #sched} {
  %0 = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr
  %2 = lsir.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr
  %c0 = arith.constant {sched.stage = 4 : i32} 0 : i32
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %dest_res_0, %token_1 = amdgcn.load global_load_dword dest %2 addr %0 {sched.stage = 1 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 {sched.stage = 0 : i32} : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait lgkm_cnt 0 {sched.stage = 0 : i32}
  amdgcn.s_barrier {sched.stage = 0 : i32}
  %dest_res_2, %token_3 = amdgcn.load ds_read_b32 dest %2 addr %1 {sched.stage = 0 : i32} : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL:   func.func @advanced_sched() {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_1:.*]] = amdgcn.v_add_u32 outs(%[[ALLOCA_1]]) ins(%[[VAL_0]], %[[VAL_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[VAL_2:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_0]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[VAL_0]] addr %[[ALLOCA_1]] : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.wait lgkm_cnt 0
// CHECK:           amdgcn.s_barrier
// CHECK:           %[[VAL_3:.*]], %[[LOAD_2:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[VAL_1]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_4:.*]] = amdgcn.v_add_i32 outs(%[[ALLOCA_1]]) ins(%[[VAL_0]], %[[VAL_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[VAL_5:.*]], %[[LOAD_3:.*]] = amdgcn.load ds_read_b32 dest %[[ALLOCA_2]] addr %[[VAL_4]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           return
// CHECK:         }
func.func @advanced_sched() attributes {
    sched = #aster_utils.generic_scheduler<#amdgcn.value_scheduler,
      #aster_utils.sched_list_labeler<[
        #amdgcn.opcode_labeler<[_s_barrier], 0>,
        #aster_utils.op_name_labeler<["arith.constant"], 4>,
        #amdgcn.opcode_labeler<[_v_add_i32], 3>,
        #amdgcn.inst_prop_labeler<[is_vmem, is_valu], 1>,
        #amdgcn.inst_prop_labeler<[dsmem], 2>
      ]>,
      #aster_utils.stage_topo_sort_sched>
  } {
  %0 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %1 = lsir.alloca : !amdgcn.vgpr
  %2 = lsir.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.load global_load_dword dest %2 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %vdst0_res = amdgcn.v_add_i32 outs(%1) ins(%dest_res, %dest_res) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  %3 = amdgcn.store ds_write_b32 data %dest_res addr %1 : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
  amdgcn.wait lgkm_cnt 0
  amdgcn.s_barrier
  %vdst0_res_0 = amdgcn.v_add_u32 outs(%1) ins(%dest_res, %dest_res) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  %dest_res_1, %token_2 = amdgcn.load ds_read_b32 dest %2 addr %vdst0_res_0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_3, %token_4 = amdgcn.load ds_read_b32 dest %2 addr %vdst0_res : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %dest_res_5, %token_6 = amdgcn.load global_load_dword dest %2 addr %0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @ssa_chain() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.v_add_u32 outs(%[[ALLOCA_2]]) ins(%[[ALLOCA_1]], %[[ALLOCA_0]]) {sched.stage = 2 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.v_add_u32 outs(%[[ALLOCA_3]]) ins(%[[ALLOCA_1]], %[[ALLOCA_0]]) {sched.stage = 1 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[VAL_2:.*]] = amdgcn.v_add_u32 outs(%[[ALLOCA_2]]) ins(%[[VAL_0]], %[[VAL_1]]) {sched.stage = 0 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           return
// CHECK:         }
// This test checks a regression in the value scheduler, the SSA chain always
// has to be respected, it doesn't matter if the value is an in or out.
func.func @ssa_chain() attributes {sched = #sched} {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %4 = amdgcn.v_add_u32 outs(%2) ins(%1, %0) {sched.stage = 2 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  %5 = amdgcn.v_add_u32 outs(%3) ins(%1, %0) {sched.stage = 1 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  %6 = amdgcn.v_add_u32 outs(%2) ins(%4, %5) {sched.stage = 0 : i32} : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  return
}
