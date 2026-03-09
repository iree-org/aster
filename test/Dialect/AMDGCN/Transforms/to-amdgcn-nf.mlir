// RUN: aster-opt --aster-to-amdgcn %s | FileCheck %s

// Verify that aster-to-amdgcn sets #amdgcn.no_reg_cast_ops post-condition
// on the amdgcn.module.

// CHECK-LABEL: amdgcn.module @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_reg_cast_ops]}
amdgcn.module @sets_postcondition target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}
