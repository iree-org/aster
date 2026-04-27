// RUN: aster-opt %s | aster-opt | FileCheck %s

// Roundtrip test for s_setprio, s_sleep, s_wakeup instructions.

// CHECK-LABEL: amdgcn.module @setprio_test
amdgcn.module @setprio_test target = #amdgcn.target<gfx942> {
  // CHECK: kernel @test_setprio_sleep_wakeup
  amdgcn.kernel @test_setprio_sleep_wakeup arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: 64, 1, 1>} {
    // CHECK: amdgcn.sopp.sopp <s_setprio>, imm = 3
    amdgcn.sopp.sopp #amdgcn.inst<s_setprio>, imm = 3
    // CHECK: amdgcn.sopp.sopp <s_setprio>
    amdgcn.sopp.sopp #amdgcn.inst<s_setprio>, imm = 0
    // CHECK: amdgcn.sopp.sopp <s_sleep>, imm = 1
    amdgcn.sopp.sopp #amdgcn.inst<s_sleep>, imm = 1
    // CHECK: amdgcn.sopp.sopp <s_wakeup>
    amdgcn.sopp.sopp #amdgcn.inst<s_wakeup>
    amdgcn.end_kernel
  }
}
