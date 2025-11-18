// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK-LABEL:Module: mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK:    .text
// CHECK:    .globl test_args
// CHECK:    .p2align 8
// CHECK:    .type test_args,@function
// CHECK:  test_args:
// CHECK:    v_mov_b32_e32 v2, 456
// CHECK:    v_mov_b32_e32 v3, v2
// CHECK:    s_endpgm
// CHECK:    .section .rodata,"a",@progbits
// CHECK:    .p2align 6, 0x0
// CHECK:    .amdhsa_kernel test_args
// CHECK:      .amdhsa_kernarg_size 56
// CHECK:      .amdhsa_user_sgpr_count 2
// CHECK:      .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:      .amdhsa_next_free_vgpr 4
// CHECK:      .amdhsa_next_free_sgpr 0
// CHECK:      .amdhsa_accum_offset 4
// CHECK:    .end_amdhsa_kernel
// CHECK:    .text
// CHECK:  .Lfunc_end0
// CHECK:    .size test_args, .Lfunc_end0-test_args
// CHECK:    .text
// CHECK:    .globl test_no_args
// CHECK:    .p2align 8
// CHECK:    .type test_no_args,@function
// CHECK:  test_no_args:
// CHECK:    s_endpgm
// CHECK:    .section .rodata,"a",@progbits
// CHECK:    .p2align 6, 0x0
// CHECK:    .amdhsa_kernel test_no_args
// CHECK:      .amdhsa_user_sgpr_count 2
// CHECK:      .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:      .amdhsa_next_free_vgpr 0
// CHECK:      .amdhsa_next_free_sgpr 0
// CHECK:      .amdhsa_accum_offset 4
// CHECK:    .end_amdhsa_kernel
// CHECK:    .text
// CHECK:  .Lfunc_end1
// CHECK:    .size test_no_args, .Lfunc_end1-test_no_args
// CHECK:    .amdgpu_metadata
// CHECK:  ---
// CHECK:  amdhsa.kernels:
// CHECK:    - .agpr_count: 0
// CHECK:      .args:
// CHECK:        - .offset: 0
// CHECK:          .size: 4
// CHECK:          .value_kind: by_value
// CHECK:        - .offset: 16
// CHECK:          .size: 12
// CHECK:          .value_kind: by_value
// CHECK:        - .access: write_only
// CHECK:          .actual_access: write_only
// CHECK:          .address_space: generic
// CHECK:          .offset: 32
// CHECK:          .size: 8
// CHECK:          .value_kind: global_buffer
// CHECK:        - .access: read_only
// CHECK:          .actual_access: read_only
// CHECK:          .address_space: private
// CHECK:          .is_const: true
// CHECK:          .is_volatile: true
// CHECK:          .offset: 40
// CHECK:          .size: 8
// CHECK:          .value_kind: global_buffer
// CHECK:        - .access: read_write
// CHECK:          .actual_access: read_write
// CHECK:          .address_space: generic
// CHECK:          .offset: 48
// CHECK:          .size: 8
// CHECK:          .value_kind: global_buffer
// CHECK:      .group_segment_fixed_size: 0
// CHECK:      .kernarg_segment_align: 16
// CHECK:      .kernarg_segment_size: 56
// CHECK:      .language: Assembler
// CHECK:      .max_flat_workgroup_size: 1024
// CHECK:      .name: test_args
// CHECK:      .private_segment_fixed_size: 0
// CHECK:      .sgpr_count: 0
// CHECK:      .sgpr_spill_count: 0
// CHECK:      .symbol: test_args.kd
// CHECK:      .vgpr_count: 2
// CHECK:      .vgpr_spill_count: 0
// CHECK:      .wavefront_size: 64
// CHECK:    - .agpr_count: 0
// CHECK:      .group_segment_fixed_size: 0
// CHECK:      .kernarg_segment_align: 8
// CHECK:      .kernarg_segment_size: 0
// CHECK:      .language: Assembler
// CHECK:      .max_flat_workgroup_size: 1024
// CHECK:      .name: test_no_args
// CHECK:      .private_segment_fixed_size: 0
// CHECK:      .sgpr_count: 0
// CHECK:      .sgpr_spill_count: 0
// CHECK:      .symbol: test_no_args.kd
// CHECK:      .vgpr_count: 0
// CHECK:      .vgpr_spill_count: 0
// CHECK:      .wavefront_size: 64
// CHECK:  amdgcn_target: amdgcn-amd-amdhsa--gfx942
// CHECK:  amdhsa.version:
// CHECK:    - 1
// CHECK:    - 2
// CHECK:  ---
// CHECK:    .end_amdgpu_metadata
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_args arguments <[
    #amdgcn.by_val_arg<size=4>,
    #amdgcn.by_val_arg<size=12, alignment=16>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>,
    #amdgcn.buffer_arg<address_space = private, access = read_only, flags = const | volatile>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write, type = !ptr.ptr<#ptr.generic_space>>
  ]> {
    %0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %1 = amdgcn.alloca : !amdgcn.vgpr<3>
    %c456 = arith.constant 456 : i32
    %2 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %c456 : (!amdgcn.vgpr<2>, i32) -> !amdgcn.vgpr<2>
    %3 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %1, %2 : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
    %4 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
  amdgcn.kernel @test_no_args {
    amdgcn.end_kernel
  }
}
