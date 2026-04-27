// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for s_mov_b32 to M0 register.

// CHECK-LABEL: Module: m0_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// CHECK-LABEL: test_s_mov_m0_imm:
// CHECK:       s_mov_b32 m0, 1024
// CHECK:       s_endpgm

// CHECK-LABEL: test_s_mov_m0_sgpr:
// CHECK:       s_mov_b32 m0, s0
// CHECK:       s_endpgm

amdgcn.module @m0_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_s_mov_m0_imm {
  ^entry:
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %c1024 = arith.constant 1024 : i32
    amdgcn.sop1 s_mov_b32 outs %m0 ins %c1024 : !amdgcn.m0<0>, i32
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_s_mov_m0_sgpr {
  ^entry:
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    amdgcn.sop1 s_mov_b32 outs %m0 ins %s0 : !amdgcn.m0<0>, !amdgcn.sgpr<0>
    amdgcn.end_kernel
  }
}
