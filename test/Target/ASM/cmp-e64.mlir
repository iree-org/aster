// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for VOPC E64 compare instructions.

// CHECK-LABEL: Module: cmp_e64_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// CHECK-LABEL: test_vcmp_eq_i32_e64:
// CHECK:       v_cmp_eq_i32_e64 s[0:1], 0, v0
// CHECK:       s_endpgm

// CHECK-LABEL: test_vcmp_eq_i32_e64_vgpr_src:
// CHECK:       v_cmp_eq_i32_e64 s[0:1], v0, v1
// CHECK:       s_endpgm

amdgcn.module @cmp_e64_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  amdgcn.kernel @test_vcmp_eq_i32_e64 {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %dst = amdgcn.make_register_range %s0, %s1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %c0 = arith.constant 0 : i32
    amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %c0, %v0
        : outs(!amdgcn.sgpr<[0 : 2]>) ins(i32, !amdgcn.vgpr<0>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_vcmp_eq_i32_e64_vgpr_src {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %dst = amdgcn.make_register_range %s0, %s1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %v0, %v1
        : outs(!amdgcn.sgpr<[0 : 2]>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
    amdgcn.end_kernel
  }
}
