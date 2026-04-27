// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: normal_forms on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_value_semantic_registers]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}

// CHECK: amdgcn.module @with_no_reg_cast target = <gfx942>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_reg_cast_ops]}
amdgcn.module @with_no_reg_cast target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_reg_cast_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<0>
    amdgcn.end_kernel
  }
}

// CHECK: amdgcn.module @without_nf target = <gfx942>
// CHECK-NOT: normal_forms
// CHECK-SAME: {
amdgcn.module @without_nf target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}
