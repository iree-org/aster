// RUN: aster-opt --pass-pipeline='builtin.module(any(amdgcn-register-coloring))' %s \
// RUN:   | FileCheck %s

// Verify that register-coloring sets the all_registers_allocated post-condition.

// CHECK-LABEL: kernel @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.all_registers_allocated]}
amdgcn.kernel @sets_postcondition {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %0 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  amdgcn.end_kernel
}
