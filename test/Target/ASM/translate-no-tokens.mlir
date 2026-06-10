// RUN: not aster-translate --split-input-file %s --mlir-to-asm 2>&1 | FileCheck %s

// CHECK: normal form violation: dependency-token operands are disallowed but found one on: amdgcn.cross_wave_token_barrier

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^entry:
    %c0 = arith.constant 0 : i32
    %addr = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<1>
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.write_token<shared>
    %fence = amdgcn.cross_wave_token_barrier deps %tok : !amdgcn.write_token<shared>
    amdgcn.end_kernel
  }
}

// -----

// CHECK: normal form violation: dependency-token operands are disallowed but found one on: amdgcn.wait

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^entry:
    %c0 = arith.constant 0 : i32
    %addr = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<1>
    %tok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.write_token<shared>
    %wf0 = amdgcn.wait deps %tok : !amdgcn.write_token<shared> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }
}
