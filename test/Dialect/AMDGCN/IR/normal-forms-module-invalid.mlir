// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Normal form violation on amdgcn.module: value-semantic vgpr inside module.
amdgcn.module @nf_vgpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  amdgcn.kernel @k {
  ^bb0:
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.end_kernel
  }
}

// -----

// Normal form violation on amdgcn.module: value-semantic sgpr.
amdgcn.module @nf_sgpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_value_semantic_registers]} {
  amdgcn.kernel @k {
  ^bb0:
    // expected-error @below {{normal form violation: register types with value semantics are disallowed but found}}
    %0 = amdgcn.alloca : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}
