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

// CHECK-LABEL: test_non_adjacent_fallthrough:
// CHECK: s_cbranch_scc0 .AMDGCN_BB_1
// CHECK: s_branch .AMDGCN_BB_2
// CHECK: .AMDGCN_BB_1:
// CHECK:   s_endpgm
// CHECK: .AMDGCN_BB_2:
// CHECK:   s_endpgm

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_cbranch_scc1 {
  ^entry:
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %scc = amdgcn.alloca : !amdgcn.scc<0>
    amdgcn.s_cmp_gt_u32 outs(%scc) ins(%s2, %s3) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<2>, !amdgcn.sgpr<3>)
    amdgcn.s_cbranch_scc1 %scc, true(^loop) false(^exit)
      : !amdgcn.scc<0>
  ^exit:
    amdgcn.end_kernel
  ^loop:
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_cbranch_scc0 {
  ^entry:
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %scc = amdgcn.alloca : !amdgcn.scc<0>
    amdgcn.s_cmp_eq_i32 outs(%scc) ins(%s0, %s1) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
    amdgcn.s_cbranch_scc0 %scc, true(^true_path) false(^false_path)
      : !amdgcn.scc<0>
  ^false_path:
    amdgcn.end_kernel
  ^true_path:
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_non_adjacent_fallthrough attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %0 = alloca : !amdgcn.scc<0>
    s_cbranch_scc0 %0, true(^bb1) false(^bb2) : !amdgcn.scc<0>
  ^bb1:  // pred: ^bb0
    end_kernel
  ^bb2:  // pred: ^bb0
    end_kernel
  }
}
