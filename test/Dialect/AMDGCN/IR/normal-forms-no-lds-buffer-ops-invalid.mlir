// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: alloc_lds in module with no_lds_buffer_ops.
amdgcn.module @has_alloc_lds target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_lds_buffer_ops]} {
  func.func @f(%arg0: index) {
    // expected-error @below {{normal form violation: LDS buffer operations are disallowed but found}}
    %0 = amdgcn.alloc_lds %arg0
    return
  }
}

// -----

// Violation: alloc_lds in kernel with no_lds_buffer_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_lds_buffer_ops]} {
  ^bb0:
    %c = arith.constant 256 : index
    // expected-error @below {{normal form violation: LDS buffer operations are disallowed but found}}
    %0 = amdgcn.alloc_lds %c
    amdgcn.end_kernel
  }
}
