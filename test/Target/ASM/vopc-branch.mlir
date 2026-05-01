// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for VOPC compare + VCC conditional branch instructions.
// Counterpart to cbranch.mlir which tests SCC-based branches.

// CHECK-LABEL: test_vcmp_lt_i32_vccnz:
// CHECK:       v_cmp_lt_i32 vcc, 0, v0
// CHECK:       s_cbranch_vccz .AMDGCN_BB_1
// CHECK:     .AMDGCN_BB_2:
// CHECK:       s_endpgm
// CHECK:     .AMDGCN_BB_1:
// CHECK:       s_endpgm

// CHECK-LABEL: test_vcmp_eq_i32_vccz:
// CHECK:       v_cmp_eq_i32 vcc, v0, v1
// CHECK:       s_cbranch_vccz .AMDGCN_BB_1
// CHECK:     .AMDGCN_BB_2:
// CHECK:       s_endpgm
// CHECK:     .AMDGCN_BB_1:
// CHECK:       s_endpgm

// CHECK-LABEL: test_vcmp_gt_swap:
// CHECK:       v_cmp_gt_i32 vcc, 32, v0
// CHECK:       s_cbranch_vccz .AMDGCN_BB_1
// CHECK:     .AMDGCN_BB_2:
// CHECK:       s_endpgm
// CHECK:     .AMDGCN_BB_1:
// CHECK:       s_endpgm

amdgcn.module @vopc_branch_mod target = #amdgcn.target<gfx942> {

  // Test 1: imm < VGPR, branch on VCC non-zero
  amdgcn.kernel @test_vcmp_lt_i32_vccnz {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    %c0 = arith.constant 0 : i32
    amdgcn.v_cmp_lt_i32 outs(%vcc) ins(%c0, %v0) : outs(!amdgcn.vcc<0>) ins(i32, !amdgcn.vgpr<0>)
    amdgcn.cbranch s_cbranch_vccz %vcc ^taken fallthrough(^fallthru)
      : !amdgcn.vcc<0>
  ^fallthru:
    amdgcn.end_kernel
  ^taken:
    amdgcn.end_kernel
  }

  // Test 2: VGPR == VGPR, branch on VCC zero
  amdgcn.kernel @test_vcmp_eq_i32_vccz {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    amdgcn.v_cmp_eq_i32 outs(%vcc) ins(%v0, %v1) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
    amdgcn.cbranch s_cbranch_vccz %vcc ^taken fallthrough(^fallthru)
      : !amdgcn.vcc<0>
  ^fallthru:
    amdgcn.end_kernel
  ^taken:
    amdgcn.end_kernel
  }

  // Test 3: Swapped operands -- v_cmp_gt_i32 with inline constant
  amdgcn.kernel @test_vcmp_gt_swap {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    %c32 = arith.constant 32 : i32
    amdgcn.v_cmp_gt_i32 outs(%vcc) ins(%c32, %v0) : outs(!amdgcn.vcc<0>) ins(i32, !amdgcn.vgpr<0>)
    amdgcn.cbranch s_cbranch_vccz %vcc ^taken fallthrough(^fallthru)
      : !amdgcn.vcc<0>
  ^fallthru:
    amdgcn.end_kernel
  ^taken:
    amdgcn.end_kernel
  }
}
