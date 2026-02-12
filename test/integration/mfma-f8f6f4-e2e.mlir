// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end kernel for scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4) on CDNA4 (gfx950).
// Fills A with FP8 E4M3 1.0 (0x38 per byte), B with FP8 E4M3 2.0 (0x40 per byte),
// C accumulator with 0.0, then stores the MFMA result.
// Identity E8M0 scales (0x7F = 127 = 2^0 = 1.0) are passed inline as scale sources.
// Expected: D[m][n] = sum_k(1.0 * 2.0) for k=0..127 = 256.0 for all elements.

// CHECK-LABEL: mfma_f8f6f4_kernel:
// CHECK:       v_mfma_ld_scale_b32
// CHECK:       v_mfma_f32_16x16x128_f8f6f4
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_f8f6f4_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr_range<[? + 4]>)
  func.func private @init_vgprx8(%cst: i32) -> (!amdgcn.vgpr_range<[? + 8]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr_range<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr_range<[? + 2]>
  }

  amdgcn.kernel @mfma_f8f6f4_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {

    // Load output pointer from kernarg
    %c_ptr = func.call @load_output_ptr()
      : () -> !amdgcn.sgpr_range<[? + 2]>

    // Thread ID (v0)
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // --- Fill A operand: 8 VGPRs with FP8 E4M3 1.0 (0x38 per byte) ---
    // FP8 E4M3: 1.0 = sign=0, exp=0111(7), mantissa=000 -> 0x38
    // Packed 4 per dword: 0x38383838 = 943208504
    %fp8_one_packed = arith.constant 943208504 : i32
    %a_range = func.call @init_vgprx8(%fp8_one_packed) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)

    // --- Fill B operand: 8 VGPRs with FP8 E4M3 2.0 (0x40 per byte) ---
    // FP8 E4M3: 2.0 = sign=0, exp=1000(8), mantissa=000 -> 0x40
    // Packed 4 per dword: 0x40404040 = 1077952576
    %fp8_two_packed = arith.constant 1077952576 : i32
    %b_range = func.call @init_vgprx8(%fp8_two_packed) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)

    // --- Initialize C accumulator: 4 VGPRs with 0 ---
    %zero_i32 = arith.constant 0 : i32
    %d_range = func.call @init_vgprx4(%zero_i32) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // --- Load identity E8M0 scale factors ---
    // E8M0: exponent 127 = 2^(127-127) = 2^0 = 1.0, packed 4 per dword: 0x7F7F7F7F
    %e8m0_identity = arith.constant 2139062143 : i32
    %scale0_s = amdgcn.alloca : !amdgcn.vgpr
    %scale0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %scale0_s, %e8m0_identity : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %scale1_s = amdgcn.alloca : !amdgcn.vgpr
    %scale1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %scale1_s, %e8m0_identity : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    // Scaled MFMA: D = A * B^T + C (16x16x128, FP8 E4M3 inputs, F32 accumulator)
    // With A=1.0 and B=2.0: D[m][n] = sum_k(1.0 * 2.0) for k=0..127 = 256.0
    // Scale sources carry identity E8M0 values (exponent=127 -> factor=1.0)
    %mfma_result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %d_range, %a_range, %b_range, %d_range, %scale0, %scale1
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    // Compute store offset: threadidx_x * 16 bytes (4 f32 values per lane)
    %offset_s = amdgcn.alloca : !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_s ins %shift_4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>

    // Store 4 f32 result values to global memory
    %c0_i32 = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dwordx4 data %mfma_result addr %c_ptr
        offset d(%thread_offset) + c(%c0_i32)
      : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
