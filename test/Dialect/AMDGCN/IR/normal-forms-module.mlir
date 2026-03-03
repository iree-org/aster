// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: normal_forms on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942> isa = <cdna3>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_value_semantic_registers]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}

// CHECK: amdgcn.module @without_nf target = <gfx942> isa = <cdna3>
// CHECK-NOT: normal_forms
// CHECK-SAME: {
amdgcn.module @without_nf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}
