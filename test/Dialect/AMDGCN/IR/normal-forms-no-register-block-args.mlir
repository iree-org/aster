// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: #amdgcn.no_register_block_args on amdgcn.module.

// CHECK: amdgcn.module @with_nf target = <gfx942>
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_register_block_args]}
amdgcn.module @with_nf target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_register_block_args]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
