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

// End-to-end test: AGPR-backed MFMA with init_agprx4.
// Loads A and B from global memory, initializes C accumulator in AGPRs to zero,
// runs MFMA, and stores AGPR result directly to global memory.

// IR check: module should have no unresolved library calls after inlining
// CHECK-LABEL: amdgcn.module
//   CHECK-NOT:   load_kernarg_pointers

// ASM check: kernel uses AGPR init, MFMA with AGPR accumulators, direct AGPR store
// ASM-LABEL: agpr_mfma_kernel:
//       ASM:   s_load_dwordx2
//       ASM:   v_accvgpr_write_b32
//       ASM:   v_accvgpr_write_b32
//       ASM:   v_accvgpr_write_b32
//       ASM:   v_accvgpr_write_b32
//       ASM:   v_mfma_f32_16x16x16_f16 a[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], a[{{[0-9]+}}:{{[0-9]+}}]
//       ASM:   global_store_dwordx4 {{.*}}, a[{{[0-9]+}}:{{[0-9]+}}]
//       ASM:   s_endpgm

// ASM: .agpr_count: 4

amdgcn.module @agpr_mfma_mod target = #amdgcn.target<gfx942> {

  // From register-init.mlir (resolved by --amdgcn-preload-library)
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_agprx4(%cst: i32) -> (!amdgcn.agpr<[? + 4]>)

  func.func private @load_kernarg_pointers() -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
    %a_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %b_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %c_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %a_ptr, %b_ptr, %c_ptr : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  }

  amdgcn.kernel @agpr_mfma_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {

    // Load kernarg pointers
    %a_ptr, %b_ptr, %c_ptr = func.call @load_kernarg_pointers()
      : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

    // v0 reserved for threadidx.x
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // Allocate A and B input register ranges (VGPRs for f16 data)
    %a_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)

    // Initialize C accumulator in AGPRs to zero
    %c0 = arith.constant 0 : i32
    %c_reg_range = func.call @init_agprx4(%c0) : (i32) -> (!amdgcn.agpr<[? + 4]>)

    // Compute per-thread offset for A/B (f16, dwordx2 = 8 bytes, shift left 3)
    %offset_a = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %c3 = arith.constant 3 : i32
    %thread_offset_f16 = amdgcn.v_lshlrev_b32 outs(%offset_a) ins(%c3, %threadidx_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr<0>)
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    // Global load A
    %loaded_a, %tok_load_a = amdgcn.global_load_dwordx2 dest %a_reg_range addr %a_ptr offset d(%thread_offset_f16) + c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>

    // Global load B
    %loaded_b, %tok_load_b = amdgcn.global_load_dwordx2 dest %b_reg_range addr %b_ptr offset d(%thread_offset_f16) + c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>

    // Wait for loads
    amdgcn.s_waitcnt vmcnt = 0

    // LDS store A
    %tok_ds_a = amdgcn.ds_write_b64 data %loaded_a addr %thread_offset_f16 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // LDS store B
    %tok_ds_b = amdgcn.ds_write_b64 data %loaded_b addr %thread_offset_f16 offset c(%c512_i32) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>

    // Wait for LDS writes
    amdgcn.s_waitcnt lgkmcnt = 0

    // LDS load A
    %loaded_a_lds, %tok_lds_a = amdgcn.ds_read_b64 dest %a_reg_range addr %thread_offset_f16 offset c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    // LDS load B
    %loaded_b_lds, %tok_lds_b = amdgcn.ds_read_b64 dest %b_reg_range addr %thread_offset_f16 offset c(%c512_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>

    // Wait for LDS reads
    amdgcn.s_waitcnt lgkmcnt = 0

    // MFMA with AGPR accumulator: C = A * B + C (AGPRs)
    %c_mfma_result = amdgcn.v_mfma_f32_16x16x16_f16 outs(%c_reg_range) ins(%loaded_a_lds, %loaded_b_lds, %c_reg_range)
    : outs(!amdgcn.agpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>)

    // Compute per-thread offset for C (f32, dwordx4 = 16 bytes, shift left 4)
    %c4 = arith.constant 4 : i32
    %thread_offset_f32 = amdgcn.v_lshlrev_b32 outs(%offset_a) ins(%c4, %threadidx_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr<0>)

    // Global store C directly from AGPRs
    %tok_store_c = amdgcn.global_store_dwordx4 data %c_mfma_result addr %c_ptr offset d(%thread_offset_f32) + c(%c0_i32) : ins(!amdgcn.agpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>

    // Wait for store
    amdgcn.s_waitcnt vmcnt = 0

    amdgcn.end_kernel
  }
}
