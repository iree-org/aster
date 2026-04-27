// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: #amdgcn.all_registers_allocated on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.all_registers_allocated]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}

// Roundtrip: both normal forms together.

// CHECK: amdgcn.module @both_nf
// CHECK-SAME: normal_forms = [#amdgcn.no_value_semantic_registers, #amdgcn.all_registers_allocated]
amdgcn.module @both_nf target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_value_semantic_registers, #amdgcn.all_registers_allocated]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
