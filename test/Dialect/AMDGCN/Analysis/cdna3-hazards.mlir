// RUN: aster-opt %s --test-hazard-analysis --split-input-file | FileCheck %s

// CDNA3 hazard analysis tests for implemented hazards.

//===----------------------------------------------------------------------===//
// Case 9: StoreHazard - Store X3/X4 -> VALU write to writedata VGPRs (2 V_NOPs)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: Symbol: cdna3_store_hazard_detected
// CHECK: Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:2, s:0, ds:0}
// CHECK:   }
func.func @cdna3_store_hazard_detected(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Case 5: VccExecVcczExeczHazard - VALU sets VCC -> VALU uses VCCZ (5 V_NOPs)
// Note: Hazard detection for compare instructions may require amdgcn.kernel context.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: Symbol: cdna3_vcc_vccz_hazard_detected
// CHECK: Op: func.func @cdna3_vcc_vccz_hazard_detected(%{{.*}}: !amdgcn.vcc<0>, %{{.*}}: !amdgcn.vccz<0>, %{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<1>, %{{.*}}: !amdgcn.vgpr<2>) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: amdgcn.v_cmp_eq_i32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_vcc_exec_vccz_execz_hazard, amdgcn.v_cmp_eq_i32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>), none, {v:5, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_cmp_eq_i32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vccz<0>, !amdgcn.vgpr<2>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_vcc_exec_vccz_execz_hazard, amdgcn.v_cmp_eq_i32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vccz<0>, !amdgcn.vgpr<2>), none, {v:5, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:5, s:0, ds:0}
// CHECK:   }
func.func @cdna3_vcc_vccz_hazard_detected(%arg0: !amdgcn.vcc<0>, %arg1: !amdgcn.vccz<0>, %arg2: !amdgcn.vgpr<0>, %arg3: !amdgcn.vgpr<1>, %arg4: !amdgcn.vgpr<2>) {
  amdgcn.v_cmp_eq_i32 outs(%arg0) ins(%arg2, %arg3) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
  amdgcn.v_cmp_eq_i32 outs(%arg0) ins(%arg1, %arg4) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vccz<0>, !amdgcn.vgpr<2>)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Case 10: ValuSgprVmemHazard - VALU writes SGPR -> VMEM reads that SGPR (5 V_NOPs)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: Symbol: cdna3_valu_sgpr_vmem_hazard_detected
// CHECK: Op: func.func @cdna3_valu_sgpr_vmem_hazard_detected(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.sgpr<0>, %{{.*}}: !amdgcn.sgpr<1>, %{{.*}}: !amdgcn.sgpr<2>) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: amdgcn.v_add_co_u32 outs(%{{.*}}, %{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<0>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_sgpr_vmem_hazard, amdgcn.v_add_co_u32 outs(%{{.*}}, %{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<0>), 1, {v:5, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = []
// CHECK:     nop counts = {v:5, s:0, ds:0}
// CHECK:   }
func.func @cdna3_valu_sgpr_vmem_hazard_detected(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.sgpr<0>, %arg2: !amdgcn.sgpr<1>, %arg3: !amdgcn.sgpr<2>) {
  %0 = amdgcn.make_register_range %arg1, %arg2 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  amdgcn.v_add_co_u32 outs(%arg0, %0) ins(%arg0, %arg0) : outs(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<0>)
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  %token = amdgcn.load global_load_dword dest %1 addr %0 : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
  return
}

// -----
//===----------------------------------------------------------------------===//
// Case 8: StoreWriteDataHazard - Store -> non-VALU write to writedata VGPRs (1 V_NOP)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: Symbol: cdna3_store_write_data_hazard_detected
// CHECK: Op: %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: %{{.*}} = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: %{{.*}} = amdgcn.make_register_range %{{.*}} : !amdgcn.vgpr<0>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.store global_store_dword data %{{.*}} addr %{{.*}} : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: %{{.*}} = amdgcn.load global_load_dword dest %{{.*}} addr %{{.*}} : dps(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = []
// CHECK:     nop counts = {v:1, s:0, ds:0}
// CHECK:   }
func.func @cdna3_store_write_data_hazard_detected(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.sgpr<0>, %arg2: !amdgcn.sgpr<1>) {
  %0 = amdgcn.make_register_range %arg1, %arg2 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  %1 = amdgcn.store global_store_dword data %arg0 addr %0 : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>
  %2 = amdgcn.alloca : !amdgcn.vgpr<0>
  %3 = amdgcn.make_register_range %2 : !amdgcn.vgpr<0>
  %token = amdgcn.load global_load_dword dest %3 addr %0 : dps(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
  return
}

// -----
//===----------------------------------------------------------------------===//
// Case 19: ValuVgprReadlaneHazard - VALU writes VGPR -> v_readfirstlane_b32
//          reads that VGPR (1 V_NOP)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: Symbol: cdna3_valu_vgpr_readlane_hazard_detected
// CHECK: Op: amdgcn.v_add_u32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard,
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard,
// CHECK:     ]
// CHECK:   }
// CHECK: Op: amdgcn.v_readfirstlane_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.vgpr<0>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     nop counts = {v:1, s:0, ds:0}
// CHECK:   }
func.func @cdna3_valu_vgpr_readlane_hazard_detected(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.sgpr<0>) {
  amdgcn.v_add_u32 outs(%arg0) ins(%arg0, %arg1) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
  amdgcn.v_readfirstlane_b32 outs(%arg2) ins(%arg0) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.vgpr<0>)
  return
}

// -----
// Case 19: No hazard when v_readfirstlane_b32 reads a different VGPR.

// CHECK-LABEL: Symbol: cdna3_valu_vgpr_readlane_no_hazard_different_vgpr
// CHECK: Op: amdgcn.v_add_u32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard,
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard,
// CHECK:     ]
// CHECK:   }
// CHECK: Op: amdgcn.v_readfirstlane_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
func.func @cdna3_valu_vgpr_readlane_no_hazard_different_vgpr(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.sgpr<0>) {
  amdgcn.v_add_u32 outs(%arg0) ins(%arg0, %arg1) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
  amdgcn.v_readfirstlane_b32 outs(%arg2) ins(%arg1) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}
