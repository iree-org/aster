// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Verify ASM emission for CDNA4 scaled f8f6f4 MFMA instructions.
// Each scaled MFMA emits a single combined v_mfma_scale_f32_*_f8f6f4 line.

// CHECK-LABEL: Module: mfma_f8f6f4_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"

// CHECK-LABEL: mfma_16x16x128:
// CHECK:       v_mfma_scale_f32_16x16x128_f8f6f4
// CHECK-SAME:  op_sel_hi:[0,0,0]
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_16x16x128_with_formats:
// CHECK:       v_mfma_scale_f32_16x16x128_f8f6f4
// CHECK-SAME:  op_sel_hi:[0,0,0] cbsz:[2] blgp:[4]
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_16x16x128_with_op_sel:
// CHECK:       v_mfma_scale_f32_16x16x128_f8f6f4
// CHECK-SAME:  op_sel_hi:[3,3,0]
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// CHECK-LABEL: mfma_32x32x64:
// CHECK:       v_mfma_scale_f32_32x32x64_f8f6f4
// CHECK-SAME:  op_sel_hi:[0,0,0]
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

amdgcn.module @mfma_f8f6f4_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr_range<[? + 4]>)
  func.func private @init_vgprx8(%cst: i32) -> (!amdgcn.vgpr_range<[? + 8]>)
  func.func private @init_vgprx16(%cst: i32) -> (!amdgcn.vgpr_range<[? + 16]>)

  func.func private @load_output_ptr() -> !amdgcn.sgpr_range<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr_range<[? + 2]>
  }

  func.func private @store_result_x4(
      %data: !amdgcn.vgpr_range<[? + 4]>) {
    %c0 = arith.constant 0 : i32
    %out = func.call @load_output_ptr() : () -> !amdgcn.sgpr_range<[? + 2]>
    %off_s = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dwordx4 data %data addr %out
        offset d(%off_s) + c(%c0)
      : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // --- 16x16x128 basic ---
  amdgcn.kernel @mfma_16x16x128 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %b = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    %s0 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- 16x16x128 with format codes ---
  amdgcn.kernel @mfma_16x16x128_with_formats arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %b = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    %s0 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    // cbsz=2 (fp6 for A), blgp=4 (fp4 for B)
    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1 cbsz = 2 blgp = 4
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- 16x16x128 with op_sel ---
  amdgcn.kernel @mfma_16x16x128_with_op_sel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %b = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %dst = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    %s0 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1 op_sel_0 = 3 op_sel_1 = 3
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // --- 32x32x64 basic ---
  amdgcn.kernel @mfma_32x32x64 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %b = func.call @init_vgprx8(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 8]>)
    %dst = func.call @init_vgprx16(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 16]>)

    %s0 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %s1 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_32x32x64_f8f6f4>
        %dst, %a, %b, %dst, %s0, %s1
        : !amdgcn.vgpr_range<[? + 8]>, !amdgcn.vgpr_range<[? + 8]>,
          !amdgcn.vgpr_range<[? + 16]>, !amdgcn.vgpr, !amdgcn.vgpr
        -> !amdgcn.vgpr_range<[? + 16]>

    // Store first 4 dwords of result to keep MFMA live
    %regs:16 = amdgcn.split_register_range %result : !amdgcn.vgpr_range<[? + 16]>
    %store_range = amdgcn.make_register_range %regs#0, %regs#1, %regs#2, %regs#3
        : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    func.call @store_result_x4(%store_range) : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    amdgcn.end_kernel
  }
}
