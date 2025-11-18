// RUN: aster-opt %s \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-register-allocation --symbol-dce \
// RUN: | FileCheck %s
//
// RUN: aster-opt %s \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-register-allocation --symbol-dce | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s --check-prefix=ASM

// CHECK-LABEL: amdgcn.module
//   CHECK-NOT:   load_kernarg_pointers

// ASM-LABEL: compute_kernel:
//       ASM:   s_load_dwordx2 s[2:3], s[0:1], 0
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  func.func private @load_kernarg_pointers() -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) {
    // Load kernarg arguments using load_arg operation
    %a_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %b_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %c_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr_range<[? + 2]>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return %a_ptr, %b_ptr, %c_ptr : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
  }

  amdgcn.kernel @compute_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {

    // Load kernarg pointers
    %a_ptr, %b_ptr, %c_ptr = func.call @load_kernarg_pointers()
      : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    // v0 reserved for threadidx.x
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // Allocate VGPRs for threadidx.x, pointers, and register placeholders
    %a_reg0 = amdgcn.alloca : !amdgcn.vgpr
    %a_reg1 = amdgcn.alloca : !amdgcn.vgpr
    %a_reg2 = amdgcn.alloca : !amdgcn.vgpr
    %a_reg3 = amdgcn.alloca : !amdgcn.vgpr
    %b_reg0 = amdgcn.alloca : !amdgcn.vgpr
    %b_reg1 = amdgcn.alloca : !amdgcn.vgpr
    %b_reg2 = amdgcn.alloca : !amdgcn.vgpr
    %b_reg3 = amdgcn.alloca : !amdgcn.vgpr
    %c_reg0 = amdgcn.alloca : !amdgcn.vgpr
    %c_reg1 = amdgcn.alloca : !amdgcn.vgpr
    %c_reg2 = amdgcn.alloca : !amdgcn.vgpr
    %c_reg3 = amdgcn.alloca : !amdgcn.vgpr

    // Initialize C registers to 0
    %c0 = arith.constant 0 : i32
    %c_reg0_init = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c_reg0, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c_reg1_init = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c_reg1, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c_reg2_init = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c_reg2, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c_reg3_init = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c_reg3, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    // Create register ranges for a_reg (quad), b_reg (quad), c_reg (quad for MFMA)
    %a_reg_range = amdgcn.make_register_range %a_reg0, %a_reg1 : !amdgcn.vgpr, !amdgcn.vgpr
    %b_reg_range = amdgcn.make_register_range %b_reg0, %b_reg1 : !amdgcn.vgpr, !amdgcn.vgpr
    %c_reg_range = amdgcn.make_register_range %c_reg0_init, %c_reg1_init, %c_reg2_init, %c_reg3_init : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // global_load (A)
    %offset_a = amdgcn.alloca : !amdgcn.vgpr
    %c3 = arith.constant 3 : i32 // shift left by dwordx2 size (8 == 2 << 3).
    %thread_offset_f16 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c3, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %loaded_a = amdgcn.flat.global_load #amdgcn.inst<global_load_dwordx2> %a_reg_range, %a_ptr[%thread_offset_f16]
      : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr] -> !amdgcn.vgpr_range<[? + 2]>

    // global_load (B)
    %loaded_b = amdgcn.flat.global_load #amdgcn.inst<global_load_dwordx2> %b_reg_range, %b_ptr[%thread_offset_f16]
      : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr] -> !amdgcn.vgpr_range<[? + 2]>

    // s_waitcnt(vmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    // // ds_store to ldsA
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %loaded_a, %thread_offset_f16, offset = %c0_i32
      : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32

    // ds_store to ldsB
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %loaded_b, %thread_offset_f16, offset = %c512_i32
      : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32

    // s_waitcnt(lgkmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // ds_load from ldsA
    %loaded_a_from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %a_reg_range, %thread_offset_f16, offset = %c0_i32
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>

    // ds_load from ldsB
    %loaded_b_from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %b_reg_range, %thread_offset_f16, offset = %c512_i32
      : !amdgcn.vgpr, i32 -> !amdgcn.vgpr_range<[? + 2]>

    // s_waitcnt(lgkmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // mfma - A and B need 2 VGPRs each, C needs 4 VGPRs
    %c_mfma_result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_reg_range, %loaded_a_from_lds, %loaded_b_from_lds, %c_reg_range
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>

    // global_store of c_mfma_result
    %c4 = arith.constant 4 : i32 // shift left by dwordx4 size (16 == 2 << 4).
    %thread_offset_f32 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx4> %c_mfma_result, %c_ptr[%thread_offset_f32]
      : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr]

    // s_waitcnt(vmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }
}
