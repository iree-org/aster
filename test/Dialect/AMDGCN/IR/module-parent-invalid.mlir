// RUN: aster-opt %s --split-input-file --verify-diagnostics

module {
  amdgcn.module @outer target = #amdgcn.target<gfx942> {
    // expected-error @+1 {{expects parent op 'builtin.module'}}
    amdgcn.module @inner target = #amdgcn.target<gfx942> {
    }
  }
}
