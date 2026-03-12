// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: index type in module with no_index_types.
amdgcn.module @has_index target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_index_types]} {
  amdgcn.kernel @k {
  ^bb0:
    // expected-error @below {{normal form violation: index types are disallowed but found}}
    %0 = arith.constant 42 : index
    amdgcn.end_kernel
  }
}

// -----

// Violation: index type in kernel with no_index_types.
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_index_types]} {
  ^bb0:
    // expected-error @below {{normal form violation: index types are disallowed but found}}
    %0 = arith.constant 42 : index
    amdgcn.end_kernel
  }
}
