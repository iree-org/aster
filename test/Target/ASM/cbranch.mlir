// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK:  ; Module: mod
// CHECK:  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK:  .text
// CHECK:  .globl test_cbranch_scc1
// CHECK:  .p2align 8
// CHECK:  .type test_cbranch_scc1,@function
// CHECK:test_cbranch_scc1:
// CHECK:  s_cmp_gt_u32 s2, s3
// CHECK:  s_cbranch_scc1 .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_2:
// CHECK:  s_endpgm
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_endpgm

// CHECK:  .text
// CHECK:  .globl test_cbranch_scc0
// CHECK:  .p2align 8
// CHECK:  .type test_cbranch_scc0,@function
// CHECK:test_cbranch_scc0:
// CHECK:  s_cmp_eq_i32 s0, s1
// CHECK:  s_cbranch_scc0 .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_2:
// CHECK:  s_endpgm
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_endpgm

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_cbranch_scc1 {
  ^entry:
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %scc = amdgcn.alloca : !amdgcn.sreg<scc, 0>
    %scc_result = amdgcn.sopc s_cmp_gt_u32 outs %scc ins %s2, %s3
      : !amdgcn.sreg<scc, 0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    amdgcn.cbranch #amdgcn.inst<s_cbranch_scc1> %scc_result ^loop fallthrough (^exit)
      : !amdgcn.sreg<scc, 0>
  ^exit:
    amdgcn.end_kernel
  ^loop:
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_cbranch_scc0 {
  ^entry:
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %scc = amdgcn.alloca : !amdgcn.sreg<scc, 0>
    %scc_result = amdgcn.sopc s_cmp_eq_i32 outs %scc ins %s0, %s1
      : !amdgcn.sreg<scc, 0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    amdgcn.cbranch #amdgcn.inst<s_cbranch_scc0> %scc_result ^true_path fallthrough (^false_path)
      : !amdgcn.sreg<scc, 0>
  ^false_path:
    amdgcn.end_kernel
  ^true_path:
    amdgcn.end_kernel
  }
}
