// Tests for option-validation in amdgcn-register-coloring. The ILP back-end is
// always available (no build-time guard), so no ilp_regalloc feature is needed.

// RUN: aster-opt --amdgcn-register-coloring="reg-alloc-solver=bad" \
// RUN:   --verify-diagnostics %s

// Verify that an unknown solver name is rejected.
amdgcn.module @bad_solver_mod target = <gfx942> {
  // expected-error @below {{reg-alloc-solver must be "greedy" or "ilp", got "bad"}}
  amdgcn.kernel @bad_solver {
    amdgcn.end_kernel
  }
}
