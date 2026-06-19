// RUN: aster-translate %s --mlir-to-asm | FileCheck %s
// RUN: aster-translate %s --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

amdgcn.module @gfx1250_vcc_select_mod target = #amdgcn.target<gfx1250> {

  // CHECK-LABEL: test_s_and_b32_vcc_lo:
  // CHECK:       s_and_b32 vcc_lo, s0, s1
  // CHECK:       s_endpgm
  amdgcn.kernel @test_s_and_b32_vcc_lo {
  ^entry:
    %vcc_lo = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %scc = amdgcn.alloca : !amdgcn.scc<0>
    amdgcn.s_and_b32 outs(%vcc_lo, %scc) ins(%s0, %s1) : outs(!amdgcn.vcc_lo<0>, !amdgcn.scc<0>) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
    amdgcn.end_kernel
  }

  // s_cselect_b32 broadcasts SCC to vcc_lo (-1 = all lanes true, 0 = all false).
  // CHECK-LABEL: test_s_cselect_b32_vcc_lo:
  // CHECK:       s_cselect_b32 vcc_lo, -1, 0
  // CHECK:       v_cndmask_b32 v2, v0, v1, vcc_lo
  // CHECK:       s_endpgm
  amdgcn.kernel @test_s_cselect_b32_vcc_lo {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %vcc_lo = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %scc = amdgcn.alloca : !amdgcn.scc<0>
    %cm1 = arith.constant -1 : i32
    %c0 = arith.constant 0 : i32
    amdgcn.s_cselect_b32 outs(%vcc_lo) ins(%cm1, %c0, %scc) : outs(!amdgcn.vcc_lo<0>) ins(i32, i32, !amdgcn.scc<0>)
    amdgcn.v_cndmask_b32 outs(%v2) ins(%v0, %v1, %vcc_lo) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vcc_lo<0>)
    amdgcn.end_kernel
  }
}
