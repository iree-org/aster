// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=0 ilp-time-limit-ms=5000})))" | FileCheck %s

!v = !amdgcn.vgpr

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // B1 scaffold: the ILP scheduler (level 0, no model yet) preserves source
  // order. Confirms the pass registers, parses its options, and runs as a
  // no-op identity reorder without disturbing the IR.
  // CHECK-LABEL: kernel @identity
  // CHECK:         alloca
  // CHECK:         alloca
  // CHECK:         alloca
  // CHECK:         end_kernel
  amdgcn.kernel @identity {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    amdgcn.end_kernel
  }
}
