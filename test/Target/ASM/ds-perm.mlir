// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for ds_permute_b32 and ds_bpermute_b32.
// Both instructions use LDS hardware for lane permutation without LDS access.

// CHECK-LABEL: Module: perm_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// CHECK-LABEL: test_permute:
// CHECK:       ds_permute_b32 v2, v0, v1
// CHECK:       s_endpgm

// CHECK-LABEL: test_permute_offset:
// CHECK:       ds_permute_b32 v2, v0, v1 offset: 8
// CHECK:       s_endpgm

// CHECK-LABEL: test_bpermute:
// CHECK:       ds_bpermute_b32 v2, v0, v1
// CHECK:       s_endpgm

// CHECK-LABEL: test_bpermute_offset:
// CHECK:       ds_bpermute_b32 v2, v0, v1 offset: 8
// CHECK:       s_endpgm

amdgcn.module @perm_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  amdgcn.kernel @test_permute {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.load ds_permute_b32 dest %v2 addr %v0
        offset d(%v1) + c(%c0)
        : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, i32)
        -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_permute_offset {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %c8 = arith.constant 8 : i32
    %tok = amdgcn.load ds_permute_b32 dest %v2 addr %v0
        offset d(%v1) + c(%c8)
        : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, i32)
        -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_bpermute {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.load ds_bpermute_b32 dest %v2 addr %v0
        offset d(%v1) + c(%c0)
        : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, i32)
        -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_bpermute_offset {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %c8 = arith.constant 8 : i32
    %tok = amdgcn.load ds_bpermute_b32 dest %v2 addr %v0
        offset d(%v1) + c(%c8)
        : dps(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, i32)
        -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }
}
