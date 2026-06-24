// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s

// Verify s_setprio, s_sleep, s_wakeup produce correct assembly.

// CHECK-LABEL: test_setprio_asm:
// CHECK: s_setprio 3
// CHECK-NEXT: s_setprio 0
// CHECK-NEXT: s_sleep 1
// CHECK-NEXT: s_wakeup
// CHECK: s_endpgm

amdgcn.module @test_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_setprio_asm arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: 64, 1, 1>} {
    s_setprio 3
    s_setprio 0
    s_sleep 1
    s_wakeup
    amdgcn.end_kernel
  }
}
