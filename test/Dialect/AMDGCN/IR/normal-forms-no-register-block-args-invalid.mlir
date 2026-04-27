// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Violation: vgpr block argument inside module with no_register_block_args.
amdgcn.module @has_reg_bbarg target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_register_block_args]} {
  // expected-error @below {{normal form violation: block arguments with register types are disallowed but found}}
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    cf.br ^bb1(%0 : !amdgcn.vgpr<3>)
  ^bb1(%arg : !amdgcn.vgpr<3>):
    amdgcn.end_kernel
  }
}

// -----

// Violation: sgpr block argument inside module with no_register_block_args.
amdgcn.module @has_sgpr_bbarg target = #amdgcn.target<gfx942> attributes {normal_forms = [#amdgcn.no_register_block_args]} {
  // expected-error @below {{normal form violation: block arguments with register types are disallowed but found}}
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.sgpr<0>
    cf.br ^bb1(%0 : !amdgcn.sgpr<0>)
  ^bb1(%arg : !amdgcn.sgpr<0>):
    amdgcn.end_kernel
  }
}

// -----

// Violation: on kernel directly.
amdgcn.module @mod target = #amdgcn.target<gfx942> {
  // expected-error @below {{normal form violation: block arguments with register types are disallowed but found}}
  amdgcn.kernel @k attributes {normal_forms = [#amdgcn.no_register_block_args]} {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<3>
    cf.br ^bb1(%0 : !amdgcn.vgpr<3>)
  ^bb1(%arg : !amdgcn.vgpr<3>):
    amdgcn.end_kernel
  }
}
