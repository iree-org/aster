// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: cf.br inside module with no_cf_branches.
amdgcn.module @has_br target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_cf_branches]} {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    // expected-error @below {{normal form violation: cf.br/cf.cond_br operations are disallowed but found}}
    cf.br ^bb1
  ^bb1:
    amdgcn.end_kernel
  }
}

// -----

// Violation: cf.cond_br inside module with no_cf_branches.
amdgcn.module @has_cond_br target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_cf_branches]} {
  amdgcn.kernel @k {
  ^bb0:
    %c = arith.constant true
    // expected-error @below {{normal form violation: cf.br/cf.cond_br operations are disallowed but found}}
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    amdgcn.end_kernel
  ^bb2:
    amdgcn.end_kernel
  }
}

// -----

// Violation: cf.br on kernel directly.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_cf_branches]} {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    // expected-error @below {{normal form violation: cf.br/cf.cond_br operations are disallowed but found}}
    cf.br ^bb1
  ^bb1:
    amdgcn.end_kernel
  }
}
