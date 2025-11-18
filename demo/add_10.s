  ; Module: add_10_module
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl add_10_kernel
  .p2align 8
  .type add_10_kernel,@function
add_10_kernel:
  v_mov_b32_e32 v11, 1
  v_mov_b32_e32 v12, 2
  v_add_u32 v10, v11, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  v_add_u32 v10, v10, v12
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel add_10_kernel
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 13
    .amdhsa_next_free_sgpr 0
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size add_10_kernel, .Lfunc_end0-add_10_kernel

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: add_10_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .sgpr_spill_count: 0
    .symbol: add_10_kernel.kd
    .vgpr_count: 3
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
