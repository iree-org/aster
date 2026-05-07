// RUN: aster-opt %s --split-input-file --verify-diagnostics

amdgcn.module @has_scf_for target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_op_with_regions]} {
  amdgcn.kernel @k {
  ^bb0:
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // expected-error @below {{normal form violation: ops with nested regions are disallowed but found}}
    scf.for %i = %c0 to %c10 step %c1 {
    }
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_op_with_regions]} {
  ^bb0:
    %c = arith.constant true
    // expected-error @below {{normal form violation: ops with nested regions are disallowed but found}}
    scf.if %c {
    }
    amdgcn.end_kernel
  }
}
