// RUN: aster-opt --pass-pipeline='builtin.module(any(amdgcn-to-register-semantics))' %s \
// RUN:   | FileCheck %s

// Verify that to-register-semantics sets the normal_forms post-condition
// on the kernel op.

// CHECK-LABEL: kernel @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_value_semantic_registers]}
amdgcn.kernel @sets_postcondition {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.end_kernel
}
