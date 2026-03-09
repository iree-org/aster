// RUN: aster-opt --pass-pipeline='builtin.module(amdgcn.module(aster-to-int-arith))' %s \
// RUN:   | FileCheck %s

// Verify that aster-to-int-arith sets the no_affine_ops post-condition on amdgcn.module.

// CHECK: amdgcn.module @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_affine_ops]}
amdgcn.module @sets_postcondition target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
