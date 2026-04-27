// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for G2S (Global-to-LDS) buffer_load with LDS flag.

// CHECK-LABEL: Module: g2s_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"

// CHECK-LABEL: test_g2s_dword:
// CHECK:       s_mov_b32 m0, 0
// CHECK:       buffer_load_dword v0, s[0:3], s4 offen lds
// CHECK:       s_endpgm

// CHECK-LABEL: test_g2s_dwordx4:
// CHECK:       s_mov_b32 m0, 0
// CHECK:       buffer_load_dwordx4 v0, s[0:3], s4 offen lds offset: 64
// CHECK:       s_endpgm

amdgcn.module @g2s_mod target = #amdgcn.target<gfx950> {

  amdgcn.kernel @test_g2s_dword {
  ^entry:
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %c0 = arith.constant 0 : i32
    amdgcn.sop1 s_mov_b32 outs %m0 ins %c0 : !amdgcn.m0<0>, i32

    // Buffer descriptor (s[0:3]) and scalar offset (s4)
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soff = amdgcn.alloca : !amdgcn.sgpr<4>
    %voff = amdgcn.alloca : !amdgcn.vgpr<0>

    %tok = amdgcn.load_lds buffer_load_dword_lds m0 %m0 addr %rsrc
        offset u(%soff) + d(%voff) + c(%c0)
        : ins(!amdgcn.m0<0>, !amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_g2s_dwordx4 {
  ^entry:
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    amdgcn.sop1 s_mov_b32 outs %m0 ins %c0 : !amdgcn.m0<0>, i32

    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soff = amdgcn.alloca : !amdgcn.sgpr<4>
    %voff = amdgcn.alloca : !amdgcn.vgpr<0>

    %tok = amdgcn.load_lds buffer_load_dwordx4_lds m0 %m0 addr %rsrc
        offset u(%soff) + d(%voff) + c(%c64)
        : ins(!amdgcn.m0<0>, !amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
