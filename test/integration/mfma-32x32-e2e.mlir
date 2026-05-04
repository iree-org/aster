// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// End-to-end test for v_mfma_f32_32x32x8_f16 on gfx942 (CDNA3).
//
// A = f16 1.0 (packed), B = f16 1.0 (packed), C = 0
// D = sum_{k=0}^{7} 1.0 * 1.0 + 0 = 8.0
//
// 32x32 MFMA uses 16 accumulator VGPRs per lane.
// Lane mapping: row = lane_id % 32, col = (lane_id / 32) * 16 + vgpr_index
// Each lane stores 16 f32 values = 64 bytes. 4 global_store_dwordx4 per lane.

// CHECK-LABEL: mfma_32x32x8_ones:
// CHECK:       v_mfma_f32_32x32x8_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_32x32_e2e_mod target = #amdgcn.target<gfx942> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx16(%cst: i32) -> (!amdgcn.vgpr<[? + 16]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  // Store 4 dwords at base + thread_offset + const_offset
  func.func private @store_x4(
      %data: !amdgcn.vgpr<[? + 4]>,
      %base: !amdgcn.sgpr<[? + 2]>,
      %thread_offset: !amdgcn.vgpr,
      %const_offset: i32) {
    %tok = amdgcn.global_store_dwordx4 data %data addr %base offset d(%thread_offset) + c(%const_offset) : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    return
  }

  // --- Kernel: A = f16 1.0, B = f16 1.0, C = 0 -> D = 8.0 ---
  // f16 1.0 = 0x3C00, packed 2 per dword: 0x3C003C00 = 1006648320
  amdgcn.kernel @mfma_32x32x8_ones arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    // A: f16 1.0 packed = 0x3C003C00 = 1006648320
    %f16_10 = arith.constant 1006648320 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %a0:2 = amdgcn.split_register_range %a : !amdgcn.vgpr<[? + 2]>
    %a0_v = amdgcn.v_mov_b32 outs(%a0#0) ins(%f16_10) : outs(!amdgcn.vgpr) ins(i32)
    %a1_v = amdgcn.v_mov_b32 outs(%a0#1) ins(%f16_10) : outs(!amdgcn.vgpr) ins(i32)
    %a_init = amdgcn.make_register_range %a0_v, %a1_v : !amdgcn.vgpr, !amdgcn.vgpr

    // B: f16 1.0 packed (same)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b0:2 = amdgcn.split_register_range %b : !amdgcn.vgpr<[? + 2]>
    %b0_v = amdgcn.v_mov_b32 outs(%b0#0) ins(%f16_10) : outs(!amdgcn.vgpr) ins(i32)
    %b1_v = amdgcn.v_mov_b32 outs(%b0#1) ins(%f16_10) : outs(!amdgcn.vgpr) ins(i32)
    %b_init = amdgcn.make_register_range %b0_v, %b1_v : !amdgcn.vgpr, !amdgcn.vgpr

    // C accumulator: 16 VGPRs initialized to zero
    %c0 = arith.constant 0 : i32
    %acc = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr<[? + 16]>)

    // MFMA: D = A * B + C
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x8_f16>
        %acc, %a_init, %b_init, %acc
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 16]> -> !amdgcn.vgpr<[? + 16]>

    // Each lane stores 16 f32 values = 64 bytes.
    // Thread offset = tid * 64 = tid << 6
    %offset_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %c6 = arith.constant 6 : i32
    %thread_offset = amdgcn.v_lshlrev_b32 outs(%offset_s) ins(%c6, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Split 16 result regs into 4 groups of 4 for dwordx4 stores
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

    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
