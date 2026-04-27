// RUN: aster-opt %s \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir" \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Verify ASM emission for v_accvgpr_write_b32, v_accvgpr_read_b32,
// init_agprx4 library, and AGPR-backed MFMA.

// CHECK-LABEL: Module: agpr_asm_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// --- Test 1: Write immediate to AGPR, read back to VGPR, store ---
// CHECK-LABEL: agpr_write_imm:
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, 42
// CHECK:       v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
// CHECK:       global_store_dword
// CHECK:       s_endpgm

// --- Test 2: Write VGPR to AGPR, read back, store ---
// CHECK-LABEL: agpr_write_vgpr:
// CHECK:       v_mov_b32_e32 v{{[0-9]+}}, 99
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
// CHECK:       v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
// CHECK:       global_store_dword
// CHECK:       s_endpgm

// --- Test 3: Init AGPRx4 with zero via library, store directly ---
// CHECK-LABEL: agpr_init_x4_zero:
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, 0
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, 0
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, 0
// CHECK:       v_accvgpr_write_b32 a{{[0-9]+}}, 0
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// --- Test 4: Init AGPRx4, use as MFMA accumulator, store ---
// CHECK-LABEL: agpr_init_mfma:
// CHECK:       v_accvgpr_write_b32
// CHECK:       v_mfma_f32_16x16x16_f16
// CHECK:       global_store_dwordx4
// CHECK:       s_endpgm

// --- Metadata: all kernels report AGPR usage ---
// CHECK:       .amdhsa_accum_offset
// CHECK:       .agpr_count:

amdgcn.module @agpr_asm_mod target = #amdgcn.target<gfx942> {

  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_agprx1() -> !amdgcn.agpr
  func.func private @alloc_vgprx2() -> !amdgcn.vgpr<[? + 2]>
  func.func private @init_agprx4(%cst: i32) -> !amdgcn.agpr<[? + 4]>

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %ptr : !amdgcn.sgpr<[? + 2]>
  }

  func.func private @store_result_x4(%data: !amdgcn.agpr<[? + 4]>) {
    %c0 = arith.constant 0 : i32
    %out = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %off = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dwordx4 data %data addr %out
        offset d(%off) + c(%c0)
      : ins(!amdgcn.agpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // Test 1: Write immediate to AGPR
  amdgcn.kernel @agpr_write_imm arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : i32

    %agpr = func.call @alloc_agprx1() : () -> !amdgcn.agpr
    %vgpr = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %agpr ins %c42 : !amdgcn.agpr, i32
    %v0 = amdgcn.vop3p v_accvgpr_read_b32 outs %vgpr ins %a0 : !amdgcn.vgpr, !amdgcn.agpr

    %out = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %off = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dword data %v0 addr %out
        offset d(%off) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  // Test 2: Write VGPR to AGPR
  amdgcn.kernel @agpr_write_vgpr arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c99 = arith.constant 99 : i32
    %c0 = arith.constant 0 : i32

    %vgpr_src = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %agpr = func.call @alloc_agprx1() : () -> !amdgcn.agpr
    %vgpr_dst = func.call @alloc_vgpr() : () -> !amdgcn.vgpr

    %v_src = amdgcn.vop1.vop1 <v_mov_b32_e32> %vgpr_src, %c99 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %a0 = amdgcn.vop3p v_accvgpr_write_b32 outs %agpr ins %v_src : !amdgcn.agpr, !amdgcn.vgpr
    %v0 = amdgcn.vop3p v_accvgpr_read_b32 outs %vgpr_dst ins %a0 : !amdgcn.vgpr, !amdgcn.agpr

    %out = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %off = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dword data %v0 addr %out
        offset d(%off) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  // Test 3: Init AGPRx4 with zero, store directly
  amdgcn.kernel @agpr_init_x4_zero arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %acc = func.call @init_agprx4(%c0) : (i32) -> !amdgcn.agpr<[? + 4]>
    func.call @store_result_x4(%acc) : (!amdgcn.agpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }

  // Test 4: Init AGPRx4 with zero, use as MFMA accumulator, store result
  amdgcn.kernel @agpr_init_mfma arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %c0 = arith.constant 0 : i32
    %a = func.call @alloc_vgprx2() : () -> !amdgcn.vgpr<[? + 2]>
    %b = func.call @alloc_vgprx2() : () -> !amdgcn.vgpr<[? + 2]>
    %acc = func.call @init_agprx4(%c0) : (i32) -> !amdgcn.agpr<[? + 4]>

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
        %acc, %a, %b, %acc
        : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>,
          !amdgcn.agpr<[? + 4]> -> !amdgcn.agpr<[? + 4]>

    func.call @store_result_x4(%result) : (!amdgcn.agpr<[? + 4]>) -> ()
    amdgcn.end_kernel
  }
}
