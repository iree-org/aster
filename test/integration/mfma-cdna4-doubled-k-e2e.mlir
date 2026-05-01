// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end tests for CDNA4 doubled-K MFMA instructions on gfx950.
//
// Kernel 1 (mfma_16x16x32_f16_ones):
//   A = f16 1.0 (packed 0x3C003C00), B = f16 1.0, C = 0
//   D = sum_{k=0}^{31} 1.0 * 1.0 = 32.0
//
// Kernel 2 (mfma_16x16x32_bf16_ones):
//   A = bf16 1.0 (packed 0x3F803F80), B = bf16 1.0, C = 0
//   D = sum_{k=0}^{31} 1.0 * 1.0 = 32.0
//
// Kernel 3 (mfma_32x32x16_bf16_ones):
//   A = bf16 1.0 (packed 0x3F803F80), B = bf16 1.0, C = 0
//   D = sum_{k=0}^{15} 1.0 * 1.0 = 16.0

// CHECK-LABEL: mfma_16x16x32_f16_ones:
// CHECK:       v_mfma_f32_16x16x32_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_16x16x32_bf16_ones:
// CHECK:       v_mfma_f32_16x16x32_bf16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_32x32x16_bf16_ones:
// CHECK:       v_mfma_f32_32x32x16_bf16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_32x32x16_f16_ones:
// CHECK:       v_mfma_f32_32x32x16_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_cdna4_doubled_k_mod target = #amdgcn.target<gfx950> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)
  // init_vgprx8 not needed -- 32x32x16 uses 4 VGPRs for A/B
  func.func private @init_vgprx16(%cst: i32) -> (!amdgcn.vgpr<[? + 16]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  func.func private @store_x4(
      %data: !amdgcn.vgpr<[? + 4]>,
      %base: !amdgcn.sgpr<[? + 2]>,
      %thread_offset: !amdgcn.vgpr,
      %const_offset: i32) {
    %tok = amdgcn.store global_store_dwordx4 data %data addr %base
        offset d(%thread_offset) + c(%const_offset)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    return
  }

  // --- Kernel 1: v_mfma_f32_16x16x32_f16 with A=B=1.0, C=0 -> D=32.0 ---
  // f16 1.0 = 0x3C00, packed 2 per dword: 0x3C003C00 = 1006648320
  amdgcn.kernel @mfma_16x16x32_f16_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    %f16_10 = arith.constant 1006648320 : i32
    %a = func.call @init_vgprx4(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @init_vgprx4(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %c0 = arith.constant 0 : i32
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_f16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%shift_4, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 2: v_mfma_f32_16x16x32_bf16 with A=B=1.0, C=0 -> D=32.0 ---
  // bf16 1.0 = 0x3F80, packed 2 per dword: 0x3F803F80 = 1065369472
  amdgcn.kernel @mfma_16x16x32_bf16_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    %bf16_10 = arith.constant 1065369472 : i32
    %a = func.call @init_vgprx4(%bf16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @init_vgprx4(%bf16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %c0 = arith.constant 0 : i32
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %shift_4 = arith.constant 4 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%shift_4, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %result addr %c_ptr
        offset d(%thread_offset) + c(%c0_store)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 3: v_mfma_f32_32x32x16_bf16 with A=B=1.0, C=0 -> D=16.0 ---
  // bf16 1.0 = 0x3F80, packed 2 per dword: 0x3F803F80 = 1065369472
  // 32x32 MFMA: 4 VGPRs per A/B (8 packed bf16), 16 VGPRs per C/D.
  // Each lane stores 16 f32 = 64 bytes. 4 x global_store_dwordx4.
  amdgcn.kernel @mfma_32x32x16_bf16_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    %bf16_10 = arith.constant 1065369472 : i32
    %a = func.call @init_vgprx4(%bf16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @init_vgprx4(%bf16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %c0 = arith.constant 0 : i32
    %acc = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr<[? + 16]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x16_bf16>
        %acc, %a, %b, %acc
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 16]> -> !amdgcn.vgpr<[? + 16]>

    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %c6 = arith.constant 6 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%c6, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %r:16 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 16]>

    %g0 = amdgcn.make_register_range %r#0, %r#1, %r#2, %r#3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g1 = amdgcn.make_register_range %r#4, %r#5, %r#6, %r#7
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g2 = amdgcn.make_register_range %r#8, %r#9, %r#10, %r#11
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g3 = amdgcn.make_register_range %r#12, %r#13, %r#14, %r#15
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %off0 = arith.constant 0 : i32
    %off16 = arith.constant 16 : i32
    %off32 = arith.constant 32 : i32
    %off48 = arith.constant 48 : i32

    func.call @store_x4(%g0, %c_ptr, %thread_offset, %off0)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g1, %c_ptr, %thread_offset, %off16)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g2, %c_ptr, %thread_offset, %off32)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g3, %c_ptr, %thread_offset, %off48)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // --- Kernel 4: v_mfma_f32_32x32x16_f16 with A=B=1.0, C=0 -> D=16.0 ---
  // f16 1.0 = 0x3C00, packed 2 per dword: 0x3C003C00 = 1006648320
  // 32x32 MFMA: 4 VGPRs per A/B (8 packed f16), 16 VGPRs per C/D.
  amdgcn.kernel @mfma_32x32x16_f16_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    %f16_10 = arith.constant 1006648320 : i32
    %a = func.call @init_vgprx4(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @init_vgprx4(%f16_10) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %c0 = arith.constant 0 : i32
    %acc = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr<[? + 16]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x16_f16>
        %acc, %a, %b, %acc
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 16]> -> !amdgcn.vgpr<[? + 16]>

    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %c6 = arith.constant 6 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%c6, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %r:16 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 16]>

    %g0 = amdgcn.make_register_range %r#0, %r#1, %r#2, %r#3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g1 = amdgcn.make_register_range %r#4, %r#5, %r#6, %r#7
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g2 = amdgcn.make_register_range %r#8, %r#9, %r#10, %r#11
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %g3 = amdgcn.make_register_range %r#12, %r#13, %r#14, %r#15
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %off0 = arith.constant 0 : i32
    %off16 = arith.constant 16 : i32
    %off32 = arith.constant 32 : i32
    %off48 = arith.constant 48 : i32

    func.call @store_x4(%g0, %c_ptr, %thread_offset, %off0)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g1, %c_ptr, %thread_offset, %off16)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g2, %c_ptr, %thread_offset, %off32)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()
    func.call @store_x4(%g3, %c_ptr, %thread_offset, %off48)
      : (!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> ()

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
