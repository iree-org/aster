// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: #amdgcn.no_metadata_ops on amdgcn.kernel.

// CHECK: kernel @k
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_metadata_ops]}
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_metadata_ops]} {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
