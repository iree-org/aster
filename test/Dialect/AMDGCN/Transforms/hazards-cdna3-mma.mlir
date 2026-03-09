// RUN: aster-opt %s --amdgcn-hazards --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @mma_nondlops_valu_mfma_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<16>) {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.vgpr<[4 : 8]>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG4]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<16>) -> ()
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[SPLIT_REGISTER_RANGE_0]]#0, %[[SPLIT_REGISTER_RANGE_0]]#1, %[[SPLIT_REGISTER_RANGE_0]]#2, %[[SPLIT_REGISTER_RANGE_0]]#3 : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[MAKE_REGISTER_RANGE_0]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           return
// CHECK:         }
func.func @mma_nondlops_valu_mfma_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<16>) {
  %0:4 = amdgcn.split_register_range %arg2 : !amdgcn.vgpr<[4 : 8]>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0#0, %arg4 : (!amdgcn.vgpr<4>, !amdgcn.vgpr<16>) -> ()
  %1 = amdgcn.make_register_range %0#0, %0#1, %0#2, %0#3 : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %1 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  return
}

// CHECK-LABEL:   func.func @mma_nondlops_valu_mfma_srca_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG5:.*]]: !amdgcn.vgpr<16>) {
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG0]], %[[ARG5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<16>) -> ()
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG4]], %[[MAKE_REGISTER_RANGE_0]], %[[ARG2]], %[[ARG3]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           return
// CHECK:         }
func.func @mma_nondlops_valu_mfma_srca_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<[2 : 4]>, %arg3: !amdgcn.vgpr<[4 : 8]>, %arg4: !amdgcn.vgpr<[8 : 12]>, %arg5: !amdgcn.vgpr<16>) {
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg0, %arg5 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<16>) -> ()
  %0 = amdgcn.make_register_range %arg0, %arg1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg4, %0, %arg2, %arg3 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_xdl_read_srcc_exact_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[12 : 16]>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG4]], %[[ARG0]], %[[ARG1]], %[[ARG3]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[8 : 12]> -> !amdgcn.vgpr<[12 : 16]>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_xdl_read_srcc_exact_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<[12 : 16]>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg4, %arg0, %arg1, %arg3 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[8 : 12]> -> !amdgcn.vgpr<[12 : 16]>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_vmem_valu_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<20>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG4]], %[[SPLIT_REGISTER_RANGE_0]]#0 : (!amdgcn.vgpr<20>, !amdgcn.vgpr<8>) -> ()
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_vmem_valu_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<20>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg4, %0#0 : (!amdgcn.vgpr<20>, !amdgcn.vgpr<8>) -> ()
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_vmem_load_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.sgpr<[0 : 2]>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[SPLIT_REGISTER_RANGE_0]]#0 addr %[[ARG4]] : dps(!amdgcn.vgpr<8>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_vmem_load_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.sgpr<[0 : 2]>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %token = amdgcn.load global_load_dword dest %0#0 addr %arg4 : dps(!amdgcn.vgpr<8>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_valu_write_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<20>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG4]] : (!amdgcn.vgpr<8>, !amdgcn.vgpr<20>) -> ()
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_valu_write_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<20>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0#0, %arg4 : (!amdgcn.vgpr<8>, !amdgcn.vgpr<20>) -> ()
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_ds_load_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<20>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[SPLIT_REGISTER_RANGE_0]]#0 addr %[[ARG4]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<8>) ins(!amdgcn.vgpr<20>, i32) -> !amdgcn.read_token<shared>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_ds_load_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<20>) {
  %c0_i32 = arith.constant 0 : i32
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %token = amdgcn.load ds_read_b32 dest %0#0 addr %arg4 offset c(%c0_i32) : dps(!amdgcn.vgpr<8>) ins(!amdgcn.vgpr<20>, i32) -> !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_global_store_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.sgpr<[0 : 2]>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[SPLIT_REGISTER_RANGE_0]]#0 addr %[[ARG4]] : ins(!amdgcn.vgpr<8>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_global_store_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.sgpr<[0 : 2]>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %1 = amdgcn.store global_store_dword data %0#0 addr %arg4 : ins(!amdgcn.vgpr<8>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_buffer_load_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.sgpr<[0 : 4]>,
// CHECK-SAME:      %[[ARG5:.*]]: !amdgcn.sgpr<4>,
// CHECK-SAME:      %[[ARG6:.*]]: !amdgcn.vgpr<20>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           %[[LOAD_0:.*]] = amdgcn.load buffer_load_dword dest %[[SPLIT_REGISTER_RANGE_0]]#0 addr %[[ARG4]] offset u(%[[ARG5]]) + d(%[[ARG6]]) + c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<8>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<20>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_buffer_load_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.sgpr<[0 : 4]>, %arg5: !amdgcn.sgpr<4>, %arg6: !amdgcn.vgpr<20>) {
  %c0_i32 = arith.constant 0 : i32
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %token = amdgcn.load buffer_load_dword dest %0#0 addr %arg4 offset u(%arg5) + d(%arg6) + c(%c0_i32) : dps(!amdgcn.vgpr<8>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<20>, i32) -> !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_ds_store_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<20>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[SPLIT_REGISTER_RANGE_0]]#0 addr %[[ARG4]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<8>, !amdgcn.vgpr<20>, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_ds_store_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<20>) {
  %c0_i32 = arith.constant 0 : i32
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %1 = amdgcn.store ds_write_b32 data %0#0 addr %arg4 offset c(%c0_i32) : ins(!amdgcn.vgpr<8>, !amdgcn.vgpr<20>, i32) -> !amdgcn.write_token<shared>
  return
}

// CHECK-LABEL:   func.func @mma_xdl_write_mfma_read_srca(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[12 : 16]>,
// CHECK-SAME:      %[[ARG5:.*]]: !amdgcn.vgpr<[16 : 20]>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[8 : 12]>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[SPLIT_REGISTER_RANGE_0]]#0, %[[SPLIT_REGISTER_RANGE_0]]#1 : !amdgcn.vgpr<8>, !amdgcn.vgpr<9>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG4]], %[[MAKE_REGISTER_RANGE_0]], %[[ARG1]], %[[ARG5]] : <[8 : 10]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 20]> -> !amdgcn.vgpr<[12 : 16]>
// CHECK:           return
// CHECK:         }
func.func @mma_xdl_write_mfma_read_srca(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<[12 : 16]>, %arg5: !amdgcn.vgpr<[16 : 20]>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  %0:4 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[8 : 12]>
  %1 = amdgcn.make_register_range %0#0, %0#1 : !amdgcn.vgpr<8>, !amdgcn.vgpr<9>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg4, %1, %arg1, %arg5 : <[8 : 10]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 20]> -> !amdgcn.vgpr<[12 : 16]>
  return
}

// CHECK-LABEL:   func.func @mma_nondlops_valu_mfma_srcb_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<2>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<3>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG5:.*]]: !amdgcn.vgpr<16>) {
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG1]], %[[ARG5]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<16>) -> ()
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG4]], %[[ARG0]], %[[MAKE_REGISTER_RANGE_0]], %[[ARG3]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           return
// CHECK:         }
func.func @mma_nondlops_valu_mfma_srcb_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<2>, %arg2: !amdgcn.vgpr<3>, %arg3: !amdgcn.vgpr<[4 : 8]>, %arg4: !amdgcn.vgpr<[8 : 12]>, %arg5: !amdgcn.vgpr<16>) {
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg1, %arg5 : (!amdgcn.vgpr<2>, !amdgcn.vgpr<16>) -> ()
  %0 = amdgcn.make_register_range %arg1, %arg2 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg4, %arg0, %0, %arg3 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  return
}

// CHECK-LABEL:   func.func @mma_valu_then_mfma_chain(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 8]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[8 : 12]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[12 : 16]>,
// CHECK-SAME:      %[[ARG5:.*]]: !amdgcn.vgpr<16>) {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:4 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.vgpr<[4 : 8]>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG5]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<16>) -> ()
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[SPLIT_REGISTER_RANGE_0]]#0, %[[SPLIT_REGISTER_RANGE_0]]#1, %[[SPLIT_REGISTER_RANGE_0]]#2, %[[SPLIT_REGISTER_RANGE_0]]#3 : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[MAKE_REGISTER_RANGE_0]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ARG4]], %[[ARG0]], %[[ARG1]], %[[ARG3]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[8 : 12]> -> !amdgcn.vgpr<[12 : 16]>
// CHECK:           return
// CHECK:         }
func.func @mma_valu_then_mfma_chain(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[4 : 8]>, %arg3: !amdgcn.vgpr<[8 : 12]>, %arg4: !amdgcn.vgpr<[12 : 16]>, %arg5: !amdgcn.vgpr<16>) {
  %0:4 = amdgcn.split_register_range %arg2 : !amdgcn.vgpr<[4 : 8]>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0#0, %arg5 : (!amdgcn.vgpr<4>, !amdgcn.vgpr<16>) -> ()
  %1 = amdgcn.make_register_range %0#0, %0#1, %0#2, %0#3 : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg3, %arg0, %arg1, %1 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[8 : 12]>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %arg4, %arg0, %arg1, %arg3 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[8 : 12]> -> !amdgcn.vgpr<[12 : 16]>
  return
}

// -----

// 32x32x8 MFMA is 8-pass (case 2): VALU read of MFMA result needs 11 NOPs.
// CHECK-LABEL:   func.func @mma_32x32x8_xdl_write_valu_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[16 : 32]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[32 : 48]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<48>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 32]> -> !amdgcn.vgpr<[32 : 48]>
// CHECK:           %[[SPLIT:.*]]:16 = amdgcn.split_register_range %[[ARG3]] : !amdgcn.vgpr<[32 : 48]>
// CHECK-COUNT-11:  amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG4]], %[[SPLIT]]#0 : (!amdgcn.vgpr<48>, !amdgcn.vgpr<32>) -> ()
// CHECK:           return
// CHECK:         }
func.func @mma_32x32x8_xdl_write_valu_hazard(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[16 : 32]>, %arg3: !amdgcn.vgpr<[32 : 48]>, %arg4: !amdgcn.vgpr<48>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 32]> -> !amdgcn.vgpr<[32 : 48]>
  %0:16 = amdgcn.split_register_range %arg3 : !amdgcn.vgpr<[32 : 48]>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg4, %0#0 : (!amdgcn.vgpr<48>, !amdgcn.vgpr<32>) -> ()
  return
}

// -----

// 32x32x8 MFMA chained: SrcC exactly same -> 0 NOPs needed (case 2).
// CHECK-LABEL:   func.func @mma_32x32x8_xdl_write_xdl_read_srcc_exact(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[0 : 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[2 : 4]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[16 : 32]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<[32 : 48]>,
// CHECK-SAME:      %[[ARG4:.*]]: !amdgcn.vgpr<[48 : 64]>) {
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %[[ARG3]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 32]> -> !amdgcn.vgpr<[32 : 48]>
// CHECK-NOT:       amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %[[ARG4]], %[[ARG0]], %[[ARG1]], %[[ARG3]] : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[32 : 48]> -> !amdgcn.vgpr<[48 : 64]>
// CHECK:           return
// CHECK:         }
func.func @mma_32x32x8_xdl_write_xdl_read_srcc_exact(%arg0: !amdgcn.vgpr<[0 : 2]>, %arg1: !amdgcn.vgpr<[2 : 4]>, %arg2: !amdgcn.vgpr<[16 : 32]>, %arg3: !amdgcn.vgpr<[32 : 48]>, %arg4: !amdgcn.vgpr<[48 : 64]>) {
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %arg3, %arg0, %arg1, %arg2 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[16 : 32]> -> !amdgcn.vgpr<[32 : 48]>
  amdgcn.vop3p.vop3p_mai <v_mfma_f32_32x32x8_f16> %arg4, %arg0, %arg1, %arg3 : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[32 : 48]> -> !amdgcn.vgpr<[48 : 64]>
  return
}
