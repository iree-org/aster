// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end tests for scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4) on CDNA4.
//
// Kernel 1 (scaled_mfma_nonidentity_scales):
//   A = FP8 E4M3 1.75 (0x3E), B = FP8 E4M3 1.25 (0x3A), C = 0
//   scale_A = 2^(129-127) = 4.0, scale_B = 2^(128-127) = 2.0
//   Expected: 4.0 * 2.0 * 128 * 1.75 * 1.25 = 2240.0
//
// Kernel 2 (scaled_mfma_split_k_accum):
//   A[k=0..63] = 1.0, A[k=64..127] = 1.5
//   B[k=0..63] = 2.0, B[k=64..127] = 0.5
//   C = 10.0 (f32), scales = identity (1.0)
//   Expected: 64*1.0*2.0 + 64*1.5*0.5 + 10.0 = 128 + 48 + 10 = 186.0

// CHECK-LABEL: scaled_mfma_nonidentity_scales:
// CHECK:       v_mfma_scale_f32_16x16x128_f8f6f4
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: scaled_mfma_split_k_accum:
// CHECK:       v_mfma_scale_f32_16x16x128_f8f6f4
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_f8f6f4_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr_range<[? + 4]>)
  func.func private @init_vgprx8(%cst: i32) -> (!amdgcn.vgpr_range<[? + 8]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr_range<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr_range<[? + 2]>
  }

  // --- Kernel 1: Non-identity scales + non-trivial FP8 values ---
  // A = 1.75 (FP8 E4M3 0x3E), B = 1.25 (FP8 E4M3 0x3A), C = 0
  // scale_A = 4.0 (E8M0 exp 129 = 0x81), scale_B = 2.0 (E8M0 exp 128 = 0x80)
  // D = scale_A * scale_B * sum_k(A*B) + C = 4 * 2 * 128 * 1.75 * 1.25 = 2240.0
  amdgcn.kernel @scaled_mfma_nonidentity_scales arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    // Pin thread ID (v0) early so allocator avoids v0 for operand ranges
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr_range<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A: FP8 E4M3 1.75 = 0x3E, packed 4 per dword: 0x3E3E3E3E = 1044266558
    %fp8_175 = arith.constant 1044266558 : i32
    %a = func.call @init_vgprx8(%fp8_175) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)

    // B: FP8 E4M3 1.25 = 0x3A, packed 4 per dword: 0x3A3A3A3A = 976894522
    %fp8_125 = arith.constant 976894522 : i32
    %b = func.call @init_vgprx8(%fp8_125) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)

    // C accumulator: zero
    %c0 = arith.constant 0 : i32
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // Scale A: E8M0 exponent 129 -> 2^(129-127) = 4.0
    // Packed: 0x81818181 = 2172748161
    %e8m0_4x = arith.constant 2172748161 : i32
    %s0_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %s0_s, %e8m0_4x : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    // Scale B: E8M0 exponent 128 -> 2^(128-127) = 2.0
    // Packed: 0x80808080 = 2155905152
    %e8m0_2x = arith.constant 2155905152 : i32
    %s1_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %s1_s, %e8m0_2x : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    // Store result: threadidx_x * 16 bytes (4 f32 per lane)
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_s ins %shift_4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 2: Split k-range + non-zero accumulator ---
  // A[k=0..63] = 1.0 (0x38), A[k=64..127] = 1.5 (0x3C)
  // B[k=0..63] = 2.0 (0x40), B[k=64..127] = 0.5 (0x30)
  // C = 10.0 (f32 0x41200000 = 1092616192), scales = identity
  // D = 64*1.0*2.0 + 64*1.5*0.5 + 10.0 = 128 + 48 + 10 = 186.0
  amdgcn.kernel @scaled_mfma_split_k_accum arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    // Pin thread ID (v0) early so allocator avoids v0 for operand ranges
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr_range<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A lower half (k=0..63): FP8 E4M3 1.0 = 0x38, packed: 0x38383838 = 943208504
    %fp8_10 = arith.constant 943208504 : i32
    %a_lo = func.call @init_vgprx4(%fp8_10) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // A upper half (k=64..127): FP8 E4M3 1.5 = 0x3C, packed: 0x3C3C3C3C = 1010580540
    %fp8_15 = arith.constant 1010580540 : i32
    %a_hi = func.call @init_vgprx4(%fp8_15) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // Combine into 8-VGPR A range
    %a0:4 = amdgcn.split_register_range %a_lo : !amdgcn.vgpr_range<[? + 4]>
    %a1:4 = amdgcn.split_register_range %a_hi : !amdgcn.vgpr_range<[? + 4]>
    %a = amdgcn.make_register_range %a0#0, %a0#1, %a0#2, %a0#3,
                                    %a1#0, %a1#1, %a1#2, %a1#3
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr,
        !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // B lower half (k=0..63): FP8 E4M3 2.0 = 0x40, packed: 0x40404040 = 1077952576
    %fp8_20 = arith.constant 1077952576 : i32
    %b_lo = func.call @init_vgprx4(%fp8_20) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // B upper half (k=64..127): FP8 E4M3 0.5 = 0x30, packed: 0x30303030 = 808464432
    %fp8_05 = arith.constant 808464432 : i32
    %b_hi = func.call @init_vgprx4(%fp8_05) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // Combine into 8-VGPR B range
    %b0:4 = amdgcn.split_register_range %b_lo : !amdgcn.vgpr_range<[? + 4]>
    %b1:4 = amdgcn.split_register_range %b_hi : !amdgcn.vgpr_range<[? + 4]>
    %b = amdgcn.make_register_range %b0#0, %b0#1, %b0#2, %b0#3,
                                    %b1#0, %b1#1, %b1#2, %b1#3
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr,
        !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // C accumulator: 10.0 as f32 = 0x41200000 = 1092616192
    %f32_10 = arith.constant 1092616192 : i32
    %dst = func.call @init_vgprx4(%f32_10) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // Identity scales: E8M0 exponent 127 -> 2^0 = 1.0
    // Packed: 0x7F7F7F7F = 2139062143
    %e8m0_id = arith.constant 2139062143 : i32
    %s0_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %s0_s, %e8m0_id : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %s1_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %s1_s, %e8m0_id : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    // Store result: threadidx_x * 16 bytes (4 f32 per lane)
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_s ins %shift_4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
