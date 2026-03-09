// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: scf.for inside module with no_scf_ops.
amdgcn.module @has_scf_for target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_scf_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    // expected-error @below {{normal form violation: SCF dialect operations are disallowed but found}}
    scf.for %i = %c0 to %c10 step %c1 {
    }
    amdgcn.end_kernel
  }
}

// -----

// Violation: scf.if inside kernel with no_scf_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_scf_ops]} {
  ^bb0:
    %c = arith.constant true
    // expected-error @below {{normal form violation: SCF dialect operations are disallowed but found}}
    scf.if %c {
    }
    amdgcn.end_kernel
  }
}
