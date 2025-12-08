// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK:  ; Module: mod
// CHECK:  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK:  .text
// CHECK:  .globl test_branch
// CHECK:  .p2align 8
// CHECK:  .type test_branch,@function
// CHECK:test_branch:
// CHECK:  s_branch .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_endpgm
// CHECK:  .section .rodata,"a",@progbits
// CHECK:  .p2align 6, 0x0
// CHECK:  .amdhsa_kernel test_branch
// CHECK:    .amdhsa_user_sgpr_count 2
// CHECK:    .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:    .amdhsa_next_free_vgpr 0
// CHECK:    .amdhsa_next_free_sgpr 0
// CHECK:    .amdhsa_accum_offset 4
// CHECK:  .end_amdhsa_kernel
// CHECK:  .text
// CHECK:.Lfunc_end0:
// CHECK:  .size test_branch, .Lfunc_end0-test_branch
// CHECK:  .text
// CHECK:  .globl test_diamond_branch
// CHECK:  .p2align 8
// CHECK:  .type test_diamond_branch,@function
// CHECK:test_diamond_branch:
// CHECK:  s_branch .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_branch .AMDGCN_BB_2
// CHECK:.AMDGCN_BB_3:
// CHECK:  s_branch .AMDGCN_BB_2
// CHECK:.AMDGCN_BB_2:
// CHECK:  s_endpgm
// CHECK:  .section .rodata,"a",@progbits
// CHECK:  .p2align 6, 0x0
// CHECK:  .amdhsa_kernel test_diamond_branch
// CHECK:    .amdhsa_user_sgpr_count 2
// CHECK:    .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:    .amdhsa_next_free_vgpr 0
// CHECK:    .amdhsa_next_free_sgpr 0
// CHECK:    .amdhsa_accum_offset 4
// CHECK:  .end_amdhsa_kernel
// CHECK:  .text
// CHECK:.Lfunc_end1:
// CHECK:  .size test_diamond_branch, .Lfunc_end1-test_diamond_branch
// CHECK:  .amdgpu_metadata
// CHECK:---
// CHECK:amdhsa.kernels:
// CHECK:  - .agpr_count: 0
// CHECK:    .group_segment_fixed_size: 0
// CHECK:    .kernarg_segment_align: 8
// CHECK:    .kernarg_segment_size: 0
// CHECK:    .language: Assembler
// CHECK:    .max_flat_workgroup_size: 1024
// CHECK:    .name: test_branch
// CHECK:    .private_segment_fixed_size: 0
// CHECK:    .sgpr_count: 0
// CHECK:    .sgpr_spill_count: 0
// CHECK:    .symbol: test_branch.kd
// CHECK:    .vgpr_count: 0
// CHECK:    .vgpr_spill_count: 0
// CHECK:    .wavefront_size: 64
// CHECK:  - .agpr_count: 0
// CHECK:    .group_segment_fixed_size: 0
// CHECK:    .kernarg_segment_align: 8
// CHECK:    .kernarg_segment_size: 0
// CHECK:    .language: Assembler
// CHECK:    .max_flat_workgroup_size: 1024
// CHECK:    .name: test_diamond_branch
// CHECK:    .private_segment_fixed_size: 0
// CHECK:    .sgpr_count: 0
// CHECK:    .sgpr_spill_count: 0
// CHECK:    .symbol: test_diamond_branch.kd
// CHECK:    .vgpr_count: 0
// CHECK:    .vgpr_spill_count: 0
// CHECK:    .wavefront_size: 64
// CHECK:amdgcn_target: amdgcn-amd-amdhsa--gfx942
// CHECK:amdhsa.version:
// CHECK:  - 1
// CHECK:  - 2
// CHECK:  ---
// CHECK:.end_amdgpu_metadata
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_branch {
  ^entry:
    amdgcn.branch #amdgcn.inst<s_branch> ^next
  ^next:
    amdgcn.end_kernel
  }
  amdgcn.kernel @test_diamond_branch {
  ^entry:
    amdgcn.branch #amdgcn.inst<s_branch> ^true
  ^true:
    amdgcn.branch #amdgcn.inst<s_branch> ^exit
  ^false:
    amdgcn.branch #amdgcn.inst<s_branch> ^exit
  ^exit:
    amdgcn.end_kernel
  }
}
