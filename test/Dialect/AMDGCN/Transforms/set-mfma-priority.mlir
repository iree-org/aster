// RUN: aster-opt %s --aster-set-mfma-priority --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @contiguous_mfma_group
//       CHECK:   amdgcn.sopp.sopp <s_setprio>, imm = 1
//  CHECK-NEXT:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//       CHECK:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//       CHECK:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//  CHECK-NEXT:   amdgcn.sopp.sopp <s_setprio>
//  CHECK-NEXT:   return
func.func @contiguous_mfma_group(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  %r0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %c
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  %r1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %r0
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  %r2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %r1
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  return
}

// -----

// CHECK-LABEL: func.func @interleaved_mfma_store
//       CHECK:   amdgcn.sopp.sopp <s_setprio>, imm = 1
//  CHECK-NEXT:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//       CHECK:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//  CHECK-NEXT:   amdgcn.sopp.sopp <s_setprio>
//       CHECK:   amdgcn.store global_store_dword
//       CHECK:   amdgcn.sopp.sopp <s_setprio>, imm = 1
//  CHECK-NEXT:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//  CHECK-NEXT:   amdgcn.sopp.sopp <s_setprio>
//       CHECK:   return
func.func @interleaved_mfma_store(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>,
    %addr: !amdgcn.vgpr<[? + 2]>, %data: !amdgcn.vgpr) {
  %r0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %c
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  %r1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %r0
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  %tok = amdgcn.store global_store_dword data %data addr %addr
    : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  %r2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %r1
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  return
}

// -----

// CHECK-LABEL: func.func @idempotent_skip
//       CHECK:   amdgcn.sopp.sopp <s_setprio>, imm = 1
//  CHECK-NEXT:   amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
//  CHECK-NEXT:   amdgcn.sopp.sopp <s_setprio>
//   CHECK-NOT:   amdgcn.sopp.sopp <s_setprio>
func.func @idempotent_skip(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  amdgcn.sopp.sopp #amdgcn.inst<s_setprio>, imm = 1
  %r0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
    %dst, %a, %b, %c
    : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  amdgcn.sopp.sopp #amdgcn.inst<s_setprio>, imm = 0
  return
}
