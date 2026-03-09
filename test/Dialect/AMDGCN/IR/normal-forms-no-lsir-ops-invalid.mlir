// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: lsir.to_reg inside module with no_lsir_ops.
amdgcn.module @has_lsir target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_lsir_ops]} {
  amdgcn.kernel @k {
  ^bb0:
    %c = arith.constant 42 : i32
    // expected-error @below {{normal form violation: LSIR dialect operations are disallowed but found}}
    %0 = lsir.to_reg %c : i32 -> !amdgcn.sgpr
    amdgcn.end_kernel
  }
}

// -----

// Violation: lsir.to_reg inside kernel with no_lsir_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_lsir_ops]} {
  ^bb0:
    %c = arith.constant 42 : i32
    // expected-error @below {{normal form violation: LSIR dialect operations are disallowed but found}}
    %0 = lsir.to_reg %c : i32 -> !amdgcn.sgpr
    amdgcn.end_kernel
  }
}
