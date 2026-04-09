// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Verify ASM emission for CDNA4 doubled-K MFMA instructions.

// CHECK-LABEL: Module: cdna4_doubled_k_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"

// CHECK-LABEL: mfma_16x16x32_f16:
// CHECK:       v_mfma_f32_16x16x32_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_16x16x32_bf16:
// CHECK:       v_mfma_f32_16x16x32_bf16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_32x32x16_bf16:
// CHECK:       v_mfma_f32_32x32x16_bf16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @cdna4_doubled_k_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx4() -> (!amdgcn.vgpr<[? + 4]>)
  func.func private @alloc_vgprx8() -> (!amdgcn.vgpr<[? + 8]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)
  func.func private @init_vgprx16(%cst: i32) -> (!amdgcn.vgpr<[? + 16]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  func.func private @store_result_x4(
      %data: !amdgcn.vgpr<[? + 4]>) {
    %c0 = arith.constant 0 : i32
    %out = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %off_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dwordx4 data %data addr %out
        offset d(%off_s) + c(%c0)
      : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // --- v_mfma_f32_16x16x32_f16 (CDNA4, 4-pass, doubled-K) ---
  amdgcn.kernel @mfma_16x16x32_f16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_f16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- v_mfma_f32_16x16x32_bf16 (CDNA4, 4-pass, doubled-K) ---
  amdgcn.kernel @mfma_16x16x32_bf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- v_mfma_f32_32x32x16_bf16 (CDNA4, 8-pass, doubled-K) ---
  amdgcn.kernel @mfma_32x32x16_bf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %b = func.call @alloc_vgprx4() : () -> (!amdgcn.vgpr<[? + 4]>)
    %dst = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr<[? + 16]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x16_bf16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>,
          !amdgcn.vgpr<[? + 16]> -> !amdgcn.vgpr<[? + 16]>

    %regs:16 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 16]>
    %store_range = amdgcn.make_register_range %regs#0, %regs#1, %regs#2, %regs#3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    func.call @store_result_x4(%store_range) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }
}
