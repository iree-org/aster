// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Verify ASM emission for 32x32 f16 MFMA instructions.

// CHECK-LABEL: Module: mfma_32x32_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// CHECK-LABEL: mfma_32x32x8_f16:
// CHECK:       v_mfma_f32_32x32x8_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_32x32_mod target = #amdgcn.target<gfx942> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx16(%cst: i32) -> (!amdgcn.vgpr<[? + 16]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
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
    amdgcn.s_waitcnt vmcnt = 0
    return
  }

  // --- v_mfma_f32_32x32x8_f16 (CDNA3+CDNA4, 16-pass) ---
  amdgcn.kernel @mfma_32x32x8_f16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %dst = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr<[? + 16]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x8_f16>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 16]> -> !amdgcn.vgpr<[? + 16]>

    // Store first 4 dwords of result to keep MFMA live
    %regs:16 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 16]>
    %store_range = amdgcn.make_register_range %regs#0, %regs#1, %regs#2, %regs#3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    func.call @store_result_x4(%store_range) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }
}
