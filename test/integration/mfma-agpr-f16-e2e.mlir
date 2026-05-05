// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end tests for AGPR-backed MFMA (v_mfma_f32_16x16x16_f16) on gfx942.
//
// Tests that AGPR init via v_accvgpr_write_b32, MFMA with AGPR accumulators,
// and direct AGPR-to-global store produce correct results.
//
// Kernel 1 (mfma_agpr_ones):
//   A = f16 1.0, B = f16 1.0, C = 0 (in AGPRs)
//   D = sum_{k=0}^{15} 1.0 * 1.0 + 0 = 16.0
//
// Kernel 2 (mfma_agpr_with_accum):
//   A = f16 1.0, B = f16 2.0, C = f32 10.0 (in AGPRs)
//   D = sum_{k=0}^{15} 1.0 * 2.0 + 10.0 = 32.0 + 10.0 = 42.0

// CHECK-LABEL: mfma_agpr_ones:
// CHECK:       v_accvgpr_write_b32
// CHECK:       v_mfma_f32_16x16x16_f16 a[{{[0-9]+}}:{{[0-9]+}}]
// CHECK:       global_store_dwordx4 {{.*}}, a[{{[0-9]+}}:{{[0-9]+}}]
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_agpr_with_accum:
// CHECK:       v_mov_b32 v{{[0-9]+}}, 1092616192
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
// CHECK:       v_mfma_f32_16x16x16_f16 a[{{[0-9]+}}:{{[0-9]+}}]
// CHECK:       global_store_dwordx4 {{.*}}, a[{{[0-9]+}}:{{[0-9]+}}]
// CHECK:       s_endpgm

// CHECK: .agpr_count:

amdgcn.module @mfma_agpr_f16_mod target = #amdgcn.target<gfx942> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @init_vgprx2(%cst: i32) -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)
  func.func private @init_agprx4(%cst: i32) -> (!amdgcn.agpr<[? + 4]>)

  // Initialize AGPRx4 from a VGPR value (for non-inline constants).
  // Moves each VGPR value to an AGPR via v_accvgpr_write_b32.
  func.func private @init_agprx4_from_vgpr(%src: !amdgcn.vgpr) -> !amdgcn.agpr<[? + 4]> {
    %r0 = amdgcn.alloca : !amdgcn.agpr
    %r1 = amdgcn.alloca : !amdgcn.agpr
    %r2 = amdgcn.alloca : !amdgcn.agpr
    %r3 = amdgcn.alloca : !amdgcn.agpr
    %a0 = amdgcn.v_accvgpr_write outs(%r0) ins(%src)
    : outs(!amdgcn.agpr) ins(!amdgcn.vgpr)
    %a1 = amdgcn.v_accvgpr_write outs(%r1) ins(%src)
    : outs(!amdgcn.agpr) ins(!amdgcn.vgpr)
    %a2 = amdgcn.v_accvgpr_write outs(%r2) ins(%src)
    : outs(!amdgcn.agpr) ins(!amdgcn.vgpr)
    %a3 = amdgcn.v_accvgpr_write outs(%r3) ins(%src)
    : outs(!amdgcn.agpr) ins(!amdgcn.vgpr)
    %range = amdgcn.make_register_range %a0, %a1, %a2, %a3
      : !amdgcn.agpr, !amdgcn.agpr, !amdgcn.agpr, !amdgcn.agpr
    return %range : !amdgcn.agpr<[? + 4]>
  }

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  // --- Kernel 1: A = f16 1.0, B = f16 1.0, C = 0 (AGPRs) -> D = 16.0 ---
  // f16 1.0 = 0x3C00, packed 2 per dword: 0x3C003C00 = 1006648320
  amdgcn.kernel @mfma_agpr_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A: f16 1.0 packed = 0x3C003C00 = 1006648320
    %f16_10 = arith.constant 1006648320 : i32
    %a = func.call @init_vgprx2(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // B: f16 1.0 packed (same)
    %b = func.call @init_vgprx2(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // C accumulator in AGPRs: zero
    %c0 = arith.constant 0 : i32
    %acc = func.call @init_agprx4(%c0) : (i32) -> (!amdgcn.agpr<[? + 4]>)

    %result = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%a, %b, %acc)
    : outs(!amdgcn.agpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>)

    // Store result directly from AGPRs: threadidx_x * 16 bytes (4 f32 per lane)
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%shift_4, %threadidx_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr<0>)
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.global_store_dwordx4 data %result addr %c_ptr offset d(%thread_offset) + c(%c0_store) : ins(!amdgcn.agpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 2: A = f16 1.0, B = f16 2.0, C = f32 10.0 (AGPRs) -> D = 42.0 ---
  // f16 2.0 = 0x4000, packed: 0x40004000 = 1073758208
  // f32 10.0 = 0x41200000 = 1092616192
  // NOTE: v_accvgpr_write_b32 only supports inline constants (0-64, special floats).
  // For literal constants like 1092616192, we first load into a VGPR via v_mov_b32,
  // then write from VGPR to AGPR.
  amdgcn.kernel @mfma_agpr_with_accum arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A: f16 1.0 packed = 0x3C003C00 = 1006648320
    %f16_10 = arith.constant 1006648320 : i32
    %a = func.call @init_vgprx2(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // B: f16 2.0 packed = 0x40004000 = 1073758208
    %f16_20 = arith.constant 1073758208 : i32
    %b = func.call @init_vgprx2(%f16_20) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // C accumulator in AGPRs: f32 10.0 via VGPR intermediate
    // (v_accvgpr_write_b32 doesn't support literal constants)
    %f32_10 = arith.constant 1092616192 : i32
    %vgpr_tmp = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %v_10 = amdgcn.v_mov_b32 outs(%vgpr_tmp) ins(%f32_10) : outs(!amdgcn.vgpr) ins(i32)
    %acc = func.call @init_agprx4_from_vgpr(%v_10) : (!amdgcn.vgpr) -> (!amdgcn.agpr<[? + 4]>)

    %result = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%a, %b, %acc)
    : outs(!amdgcn.agpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>)

    // Store result directly from AGPRs
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%shift_4, %threadidx_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr<0>)
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.global_store_dwordx4 data %result addr %c_ptr offset d(%thread_offset) + c(%c0_store) : ins(!amdgcn.agpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
