// RUN: aster-opt --pass-pipeline='builtin.module(any(amdgcn-convert-scf-control-flow))' %s \
// RUN:   | FileCheck %s

// Verify that convert-scf-control-flow sets the no_scf_ops post-condition.

// CHECK-LABEL: kernel @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_scf_ops]}
amdgcn.kernel @sets_postcondition {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr<3>
  amdgcn.end_kernel
}
