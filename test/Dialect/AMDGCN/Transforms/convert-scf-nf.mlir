// RUN: aster-opt --pass-pipeline='builtin.module(any(aster-convert-scf-control-flow))' %s \
// RUN:   | FileCheck %s

// Verify that convert-scf-control-flow converts scf.for with no remaining SCF ops.

// CHECK-LABEL: kernel @no_remaining_scf
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
amdgcn.kernel @no_remaining_scf {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr<3>
  amdgcn.end_kernel
}
