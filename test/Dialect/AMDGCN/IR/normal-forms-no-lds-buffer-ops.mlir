// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: #amdgcn.no_lds_buffer_ops on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942> isa = <cdna3>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_lds_buffer_ops]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_lds_buffer_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = arith.constant 42 : i32
    amdgcn.end_kernel
  }
}
