// RUN: aster-opt %s --aster-set-mfma-priority --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @contiguous_mfma_group
//       CHECK:   amdgcn.s_setprio 1
//  CHECK-NEXT:   amdgcn.v_mfma_f32_16x16x16_f16
//       CHECK:   amdgcn.v_mfma_f32_16x16x16_f16
//       CHECK:   amdgcn.v_mfma_f32_16x16x16_f16
//  CHECK-NEXT:   amdgcn.s_setprio 0
//  CHECK-NEXT:   return
func.func @contiguous_mfma_group(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  %r0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  %r1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %r0)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  %r2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %r1)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  return
}

// -----

// CHECK-LABEL: func.func @interleaved_mfma_store
//       CHECK:   amdgcn.s_setprio 1
//  CHECK-NEXT:   amdgcn.v_mfma_f32_16x16x16_f16
//       CHECK:   amdgcn.v_mfma_f32_16x16x16_f16
//  CHECK-NEXT:   amdgcn.s_setprio 0
//       CHECK:   amdgcn.global_store_dword
//       CHECK:   amdgcn.s_setprio 1
//  CHECK-NEXT:   amdgcn.v_mfma_f32_16x16x16_f16
//  CHECK-NEXT:   amdgcn.s_setprio 0
//       CHECK:   return
func.func @interleaved_mfma_store(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>,
    %addr: !amdgcn.vgpr<[? + 2]>, %data: !amdgcn.vgpr) {
  %r0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  %r1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %r0)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  %c0_i32_mig1 = arith.constant 0 : i32
  %tok = amdgcn.global_store_dword data %data addr %addr offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  %r2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %r1)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  return
}

// -----

// CHECK-LABEL: func.func @idempotent_skip
//       CHECK:   amdgcn.s_setprio 1
//  CHECK-NEXT:   amdgcn.v_mfma_f32_16x16x16_f16
//  CHECK-NEXT:   amdgcn.s_setprio 0
//   CHECK-NOT:   amdgcn.s_setprio
func.func @idempotent_skip(
    %a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  amdgcn.s_setprio 1
  %r0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
  amdgcn.s_setprio 0
  return
}
