// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: !aster_utils.any type in module with no_unresolved_any_types.
amdgcn.module @has_any target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> attributes {normal_forms = [#amdgcn.no_unresolved_any_types]} {
  // expected-error @below {{normal form violation: unresolved any types are disallowed but found}}
  func.func @f(%arg0: !aster_utils.any) {
    return
  }
}
