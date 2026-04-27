// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: func.call inside kernel with all_inlined (kernel-level).
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  func.func private @helper(%x: index) -> index {
    return %x : index
  }
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.all_inlined]} {
  ^bb0:
    %c0 = arith.constant 0 : index
    // expected-error @below {{normal form violation: func.call operations are disallowed (all functions should be inlined) but found call to 'helper'}}
    %r = func.call @helper(%c0) : (index) -> index
    amdgcn.end_kernel
  }
}

// -----

// Violation: func.call inside module with all_inlined (module-level).
amdgcn.module @mod target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.all_inlined]} {
  func.func private @helper(%x: index) -> index {
    return %x : index
  }
  amdgcn.kernel @k {
  ^bb0:
    %c0 = arith.constant 0 : index
    // expected-error @below {{normal form violation: func.call operations are disallowed (all functions should be inlined) but found call to 'helper'}}
    %r = func.call @helper(%c0) : (index) -> index
    amdgcn.end_kernel
  }
}
