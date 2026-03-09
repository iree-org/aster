// RUN: aster-opt --pass-pipeline='builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))' %s \
// RUN:   | FileCheck %s

// Verify that expand-md-ops correctly expands metadata ops in a kernel
// that already has the no_metadata_ops normal form (manually set).
// The pass does not auto-set this attribute because the pipeline may
// invoke it multiple times.

// CHECK: kernel @k
// CHECK-SAME: attributes {{{.*}}normal_forms = [#amdgcn.no_metadata_ops]
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_metadata_ops]} {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    amdgcn.end_kernel
  }
}
