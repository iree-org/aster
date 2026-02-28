// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end tests for CDNA3 FP8 MFMA (v_mfma_f32_16x16x32_fp8_fp8) on gfx942.
//
// NOTE: CDNA3 uses FP8 E4M3FNUZ format (bias=8), NOT OCP E4M3 (bias=7).
// Value = 2^(E-8) * (1 + M/8) for E>0, or 2^(-7) * (M/8) for E=0.
//
// Kernel 1 (mfma_fp8_ones):
//   A = FP8 E4M3FNUZ 1.0 (0x40), B = FP8 E4M3FNUZ 1.0 (0x40), C = 0
//   D = sum_{k=0}^{31} 1.0 * 1.0 + 0 = 32.0
//
// Kernel 2 (mfma_fp8_with_accum):
//   A = FP8 E4M3FNUZ 1.5 (0x44), B = FP8 E4M3FNUZ 2.0 (0x48), C = 10.0 (f32)
//   D = sum_{k=0}^{31} 1.5 * 2.0 + 10.0 = 96.0 + 10.0 = 106.0

// CHECK-LABEL: mfma_fp8_ones:
// CHECK:       v_mfma_f32_16x16x32_fp8_fp8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_fp8_with_accum:
// CHECK:       v_mfma_f32_16x16x32_fp8_fp8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_fp8_e2e_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @init_vgprx2(%cst: i32) -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  // --- Kernel 1: A = 1.0, B = 1.0, C = 0 -> D = 32.0 ---
  // FP8 E4M3FNUZ 1.0: S=0, E=8(1000), M=0(000) -> 0100 0000 = 0x40
  // Packed 4 per dword: 0x40404040 = 1077952576
  amdgcn.kernel @mfma_fp8_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A: FP8 E4M3FNUZ 1.0 = 0x40, packed: 0x40404040 = 1077952576
    %fp8_10 = arith.constant 1077952576 : i32
    %a = func.call @init_vgprx2(%fp8_10) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // B: same as A
    %b = func.call @init_vgprx2(%fp8_10) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // C accumulator: zero
    %c0 = arith.constant 0 : i32
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_fp8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    // Store result: threadidx_x * 16 bytes (4 f32 per lane)
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_s ins %shift_4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 2: A = 1.5, B = 2.0, C = 10.0 -> D = 106.0 ---
  // FP8 E4M3FNUZ 1.5: S=0, E=8(1000), M=4(100) -> 0100 0100 = 0x44
  // Packed: 0x44444444 = 1145324612
  // FP8 E4M3FNUZ 2.0: S=0, E=9(1001), M=0(000) -> 0100 1000 = 0x48
  // Packed: 0x48484848 = 1212696648
  // f32 10.0 = 0x41200000 = 1092616192
  amdgcn.kernel @mfma_fp8_with_accum arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // A: FP8 E4M3FNUZ 1.5 = 0x44, packed: 0x44444444 = 1145324612
    %fp8_15 = arith.constant 1145324612 : i32
    %a = func.call @init_vgprx2(%fp8_15) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // B: FP8 E4M3FNUZ 2.0 = 0x48, packed: 0x48484848 = 1212696648
    %fp8_20 = arith.constant 1212696648 : i32
    %b = func.call @init_vgprx2(%fp8_20) : (i32) -> (!amdgcn.vgpr<[? + 2]>)

    // C accumulator: 10.0 as f32 = 0x41200000 = 1092616192
    %f32_10 = arith.constant 1092616192 : i32
    %dst = func.call @init_vgprx4(%f32_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_fp8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    // Store result: threadidx_x * 16 bytes (4 f32 per lane)
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_s ins %shift_4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
