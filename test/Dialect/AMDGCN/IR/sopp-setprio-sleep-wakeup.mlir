// RUN: aster-opt %s | aster-opt | FileCheck %s

// Roundtrip test for s_setprio, s_sleep, s_wakeup instructions.

// CHECK-LABEL: amdgcn.module @setprio_test
amdgcn.module @setprio_test target = #amdgcn.target<gfx942> {
  // CHECK: kernel @test_setprio_sleep_wakeup
  amdgcn.kernel @test_setprio_sleep_wakeup arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: 64, 1, 1>} {
    // CHECK: s_setprio 3
    s_setprio 3
    // CHECK: s_setprio 0
    s_setprio 0
    // CHECK: s_sleep 1
    s_sleep 1
    // CHECK: s_wakeup
    s_wakeup
    amdgcn.end_kernel
  }
}
