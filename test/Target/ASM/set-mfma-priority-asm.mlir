// RUN: aster-opt %s --aster-set-mfma-priority \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s

// CHECK-LABEL: test_setprio_asm:
// CHECK: s_setprio 1
// CHECK-NEXT: s_setprio 0
// CHECK-NEXT: s_endpgm

amdgcn.module @test_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_setprio_asm arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: 64, 1, 1>} {
    s_setprio 1
    s_setprio 0
    amdgcn.end_kernel
  }
}
