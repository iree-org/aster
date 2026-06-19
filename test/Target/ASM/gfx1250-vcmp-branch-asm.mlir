// RUN: aster-translate %s --mlir-to-asm | FileCheck %s
// RUN: aster-translate %s --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// gfx1250: v_cmp -> vcc_lo, s_cbranch_vccnz consumes vcc_lo.

// CHECK-LABEL: test_vcmp_lt_i32_vcc_lo:
// CHECK:       v_cmp_lt_i32_e64 vcc_lo, 0, v0
// CHECK:       s_cbranch_vccnz .AMDGCN_BB_1
// CHECK:       s_endpgm

amdgcn.module @gfx1250_vcmp_branch_mod target = #amdgcn.target<gfx1250> {

  amdgcn.kernel @test_vcmp_lt_i32_vcc_lo {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vcc_lo = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %c0 = arith.constant 0 : i32
    amdgcn.v_cmp_lt_i32 outs(%vcc_lo) ins(%c0, %v0) : outs(!amdgcn.vcc_lo<0>) ins(i32, !amdgcn.vgpr<0>)
    amdgcn.s_cbranch_vccnz %vcc_lo, true(^taken) false(^fallthru) : !amdgcn.vcc_lo<0>
  ^fallthru:
    amdgcn.end_kernel
  ^taken:
    amdgcn.end_kernel
  }
}
