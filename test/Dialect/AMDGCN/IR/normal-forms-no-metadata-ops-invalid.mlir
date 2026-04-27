// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: amdgcn.load_arg inside kernel with no_metadata_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k
      arguments <[#amdgcn.buffer_arg<>]>
      attributes {normal_forms = [#amdgcn.no_metadata_ops]} {
  ^bb0:
    // expected-error @below {{normal form violation: AMDGCN metadata operations are disallowed but found}}
    %0 = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.end_kernel
  }
}

// -----

// Violation: amdgcn.thread_id inside kernel with no_metadata_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_metadata_ops]} {
  ^bb0:
    // expected-error @below {{normal form violation: AMDGCN metadata operations are disallowed but found}}
    %0 = amdgcn.thread_id x : !amdgcn.vgpr
    amdgcn.end_kernel
  }
}

// -----

// Violation: amdgcn.block_id inside kernel with no_metadata_ops.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_metadata_ops]} {
  ^bb0:
    // expected-error @below {{normal form violation: AMDGCN metadata operations are disallowed but found}}
    %0 = amdgcn.block_id x : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}
