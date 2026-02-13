// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for gfx950 (CDNA4) target.

// CHECK-LABEL: Module: gfx950_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
// CHECK:    .text
// CHECK:    .globl simple_kernel
// CHECK:    .p2align 8
// CHECK:    .type simple_kernel,@function
// CHECK:  simple_kernel:
// CHECK:    v_mov_b32_e32 v2, 42
// CHECK:    s_endpgm
// CHECK:    .section .rodata,"a",@progbits
// CHECK:    .p2align 6, 0x0
// CHECK:    .amdhsa_kernel simple_kernel
// CHECK:      .amdhsa_user_sgpr_count 2
// CHECK:      .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:      .amdhsa_next_free_vgpr 3
// CHECK:      .amdhsa_next_free_sgpr 0
// CHECK:      .amdhsa_accum_offset 4
// CHECK:    .end_amdhsa_kernel
// CHECK:    .text
// CHECK:  .Lfunc_end0
// CHECK:    .size simple_kernel, .Lfunc_end0-simple_kernel
// CHECK:    .text
// CHECK:    .globl empty_kernel
// CHECK:    .p2align 8
// CHECK:    .type empty_kernel,@function
// CHECK:  empty_kernel:
// CHECK:    s_endpgm
// CHECK:    .section .rodata,"a",@progbits
// CHECK:    .p2align 6, 0x0
// CHECK:    .amdhsa_kernel empty_kernel
// CHECK:      .amdhsa_user_sgpr_count 2
// CHECK:      .amdhsa_user_sgpr_kernarg_segment_ptr 1
// CHECK:      .amdhsa_next_free_vgpr 0
// CHECK:      .amdhsa_next_free_sgpr 0
// CHECK:      .amdhsa_accum_offset 4
// CHECK:    .end_amdhsa_kernel
// CHECK:    .text
// CHECK:  .Lfunc_end1
// CHECK:    .size empty_kernel, .Lfunc_end1-empty_kernel
// CHECK:    .amdgpu_metadata
// CHECK:  ---
// CHECK:  amdhsa.kernels:
// CHECK:    - .agpr_count: 0
// CHECK:      .group_segment_fixed_size: 0
// CHECK:      .kernarg_segment_align: 8
// CHECK:      .kernarg_segment_size: 0
// CHECK:      .language: Assembler
// CHECK:      .max_flat_workgroup_size: 1024
// CHECK:      .name: simple_kernel
// CHECK:      .private_segment_fixed_size: 0
// CHECK:      .sgpr_count: 0
// CHECK:      .sgpr_spill_count: 0
// CHECK:      .symbol: simple_kernel.kd
// CHECK:      .vgpr_count: 1
// CHECK:      .vgpr_spill_count: 0
// CHECK:      .wavefront_size: 64
// CHECK:    - .agpr_count: 0
// CHECK:      .group_segment_fixed_size: 0
// CHECK:      .kernarg_segment_align: 8
// CHECK:      .kernarg_segment_size: 0
// CHECK:      .language: Assembler
// CHECK:      .max_flat_workgroup_size: 1024
// CHECK:      .name: empty_kernel
// CHECK:      .private_segment_fixed_size: 0
// CHECK:      .sgpr_count: 0
// CHECK:      .sgpr_spill_count: 0
// CHECK:      .symbol: empty_kernel.kd
// CHECK:      .vgpr_count: 0
// CHECK:      .vgpr_spill_count: 0
// CHECK:      .wavefront_size: 64
// CHECK:  amdgcn_target: amdgcn-amd-amdhsa--gfx950
// CHECK:  amdhsa.version:
// CHECK:    - 1
// CHECK:    - 2
// CHECK:  ---
// CHECK:    .end_amdgpu_metadata
amdgcn.module @gfx950_mod target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {
  amdgcn.kernel @simple_kernel {
    %0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %c42 = arith.constant 42 : i32
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %c42 : (!amdgcn.vgpr<2>, i32) -> ()
    amdgcn.end_kernel
  }
  amdgcn.kernel @empty_kernel {
    amdgcn.end_kernel
  }
}
