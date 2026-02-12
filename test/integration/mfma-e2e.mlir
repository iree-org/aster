// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | FileCheck %s
//
// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s --check-prefix=ASM

// CHECK-LABEL: amdgcn.module
//   CHECK-NOT:   load_kernarg_pointers

// ASM-LABEL: compute_kernel:
//       ASM:   s_load_dwordx2 s[2:3], s[0:1], 0
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr_range<[? + 2]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr_range<[? + 4]>)

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

    // Allocate register ranges via library functions
    %a_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr_range<[? + 2]>)
    %b_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr_range<[? + 2]>)
    %c0 = arith.constant 0 : i32
    %c_reg_range = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr_range<[? + 4]>)

    // global_load (A)
    %offset_a = amdgcn.alloca : !amdgcn.vgpr
    %c3 = arith.constant 3 : i32 // shift left by dwordx2 size (8 == 2 << 3).
    %thread_offset_f16 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c3, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    %loaded_a, %tok_load_a = amdgcn.load global_load_dwordx2 dest %a_reg_range addr %a_ptr offset d(%thread_offset_f16) + c(%c0_i32) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>

    // global_load (B)
    %loaded_b, %tok_load_b = amdgcn.load global_load_dwordx2 dest %b_reg_range addr %b_ptr offset d(%thread_offset_f16) + c(%c0_i32) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>

    // s_waitcnt(vmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // // ds_store to ldsA
    %tok_ds_a = amdgcn.store ds_write_b64 data %loaded_a addr %thread_offset_f16 offset c(%c0_i32) : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // ds_store to ldsB
    %tok_ds_b = amdgcn.store ds_write_b64 data %loaded_b addr %thread_offset_f16 offset c(%c512_i32) : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // s_waitcnt(lgkmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // ds_load from ldsA
    %loaded_a_from_lds, %tok_lds_a = amdgcn.load ds_read_b64 dest %a_reg_range addr %thread_offset_f16 offset c(%c0_i32) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // ds_load from ldsB
    %loaded_b_from_lds, %tok_lds_b = amdgcn.load ds_read_b64 dest %b_reg_range addr %thread_offset_f16 offset c(%c512_i32) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // s_waitcnt(lgkmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // mfma - A and B need 2 VGPRs each, C needs 4 VGPRs
    %c_mfma_result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_reg_range, %loaded_a_from_lds, %loaded_b_from_lds, %c_reg_range
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>

    // global_store of c_mfma_result
    %c4 = arith.constant 4 : i32 // shift left by dwordx4 size (16 == 2 << 4).
    %thread_offset_f32 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %tok_store_c = amdgcn.store global_store_dwordx4 data %c_mfma_result addr %c_ptr offset d(%thread_offset_f32) + c(%c0_i32) : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<flat>

    // s_waitcnt(vmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }
}
