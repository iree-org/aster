// RUN: aster-opt %s | aster-opt | FileCheck %s
// RUN: aster-opt %s --mlir-print-op-generic | aster-opt | FileCheck %s

// Roundtrip: normal_forms on amdgcn.kernel.

// CHECK: kernel @with_nf
// CHECK-SAME: attributes {normal_forms = [#amdgcn.no_value_semantic_registers]}
amdgcn.module @test target = #amdgcn.target<gfx942> {
  amdgcn.kernel @with_nf attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.end_kernel
  }

  // CHECK: kernel @without_nf
  // CHECK-NOT: normal_forms
  // CHECK-SAME: {
  amdgcn.kernel @without_nf {
  ^bb0:
    amdgcn.end_kernel
  }
}
