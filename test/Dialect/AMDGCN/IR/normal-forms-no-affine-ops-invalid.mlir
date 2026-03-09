// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: affine.apply inside module with no_affine_ops.
amdgcn.module @has_affine target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_affine_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %c = arith.constant 42 : index
    // expected-error @below {{normal form violation: affine dialect operations are disallowed but found}}
    %0 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c)
    amdgcn.end_kernel
  }
}

// -----

// Violation: affine.apply inside kernel with no_affine_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_affine_ops]} {
  ^bb0:
    %c = arith.constant 42 : index
    // expected-error @below {{normal form violation: affine dialect operations are disallowed but found}}
    %0 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c)
    amdgcn.end_kernel
  }
}
