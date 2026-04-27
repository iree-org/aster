// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Normal form violation on amdgcn.kernel: value-semantic vgpr.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  ^bb0:
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.end_kernel
  }
}

// -----

// Normal form violation on amdgcn.kernel: value-semantic sgpr.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  ^bb0:
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}
