// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: #amdgcn.no_scf_ops on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_scf_ops]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_scf_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
