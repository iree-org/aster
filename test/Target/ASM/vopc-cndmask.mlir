// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK-LABEL: test_vcndmask_vgpr_operands:
// CHECK:       v_cndmask_b32 v2, v0, v1, vcc
// CHECK:       s_endpgm

// CHECK-LABEL: test_vcndmask_imm_true_prelude:
// CHECK:       v_mov_b32 v2, 42
// CHECK:       v_cndmask_b32 v2, v0, v2, vcc
// CHECK:       s_endpgm

// CHECK-LABEL: test_vcndmask_sgpr_true_prelude:
// CHECK:       v_mov_b32 v2, s0
// CHECK:       v_cndmask_b32 v2, v0, v2, vcc
// CHECK:       s_endpgm

amdgcn.module @vopc_cndmask_mod target = #amdgcn.target<gfx942> {

  amdgcn.kernel @test_vcndmask_vgpr_operands {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    amdgcn.v_cndmask_b32 outs(%v2) ins(%v0, %v1, %vcc) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vcc<0>)
    amdgcn.end_kernel
  }

  // True value is scalar; legalize-cf materializes it into the destination VGPR.
  amdgcn.kernel @test_vcndmask_imm_true_prelude {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    %c42 = arith.constant 42 : i32
    amdgcn.v_mov_b32 outs(%v2) ins(%c42) : outs(!amdgcn.vgpr<2>) ins(i32)
    amdgcn.v_cndmask_b32 outs(%v2) ins(%v0, %v2, %vcc) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<2>, !amdgcn.vcc<0>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_vcndmask_sgpr_true_prelude {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    amdgcn.v_mov_b32 outs(%v2) ins(%s0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.sgpr<0>)
    amdgcn.v_cndmask_b32 outs(%v2) ins(%v0, %v2, %vcc) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<2>, !amdgcn.vcc<0>)
    amdgcn.end_kernel
  }
}
