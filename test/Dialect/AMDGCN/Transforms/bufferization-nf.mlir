// RUN: aster-opt --pass-pipeline='builtin.module(any(aster-amdgcn-bufferization))' %s \
// RUN:   | FileCheck %s

// Verify that bufferization sets the no_register_block_args post-condition.

// CHECK-LABEL: kernel @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_register_block_args]}
amdgcn.kernel @sets_postcondition {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr
  amdgcn.end_kernel
}
