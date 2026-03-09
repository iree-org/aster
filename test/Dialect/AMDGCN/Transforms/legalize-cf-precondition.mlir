// RUN: aster-opt --pass-pipeline='builtin.module(any(amdgcn-legalize-cf))' %s \
// RUN:   --verify-diagnostics

// Verify that amdgcn-legalize-cf rejects kernels without all_registers_allocated.

// expected-error @below {{amdgcn-legalize-cf requires #amdgcn.all_registers_allocated normal form}}
amdgcn.kernel @missing_precondition {
^bb0:
  %0 = amdgcn.alloca : !amdgcn.vgpr<3>
  amdgcn.end_kernel
}
