// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: unallocated register (?) inside module with all_registers_allocated.
amdgcn.module @unalloc target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
  amdgcn.kernel @k {
  ^bb0:
    // expected-error @below {{normal form violation: all registers must have allocated semantics but found}}
    %0 = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.end_kernel
  }
}

// -----

// Violation: value-semantic register inside module with all_registers_allocated.
amdgcn.module @value target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
  amdgcn.kernel @k {
  ^bb0:
    // expected-error @below {{normal form violation: all registers must have allocated semantics but found}}
    %0 = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.end_kernel
  }
}

// -----

// Violation: on kernel directly.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
  ^bb0:
    // expected-error @below {{normal form violation: all registers must have allocated semantics but found}}
    %0 = amdgcn.alloca : !amdgcn.sgpr<?>
    amdgcn.end_kernel
  }
}
