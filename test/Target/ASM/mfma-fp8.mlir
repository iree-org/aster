// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Verify ASM emission for CDNA3 FP8 MFMA instructions (16x16x32).

// CHECK-LABEL: Module: mfma_fp8_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// CHECK-LABEL: mfma_fp8_fp8:
// CHECK:       v_mfma_f32_16x16x32_fp8_fp8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_fp8_bf8:
// CHECK:       v_mfma_f32_16x16x32_fp8_bf8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_bf8_fp8:
// CHECK:       v_mfma_f32_16x16x32_bf8_fp8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_bf8_bf8:
// CHECK:       v_mfma_f32_16x16x32_bf8_bf8
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_fp8_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)

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

  // --- fp8_fp8 ---
  amdgcn.kernel @mfma_fp8_fp8 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_fp8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- fp8_bf8 ---
  amdgcn.kernel @mfma_fp8_bf8 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_bf8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- bf8_fp8 ---
  amdgcn.kernel @mfma_bf8_fp8 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf8_fp8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- bf8_bf8 ---
  amdgcn.kernel @mfma_bf8_bf8 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf8_bf8>
        %dst, %a, %b, %dst
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }
}
