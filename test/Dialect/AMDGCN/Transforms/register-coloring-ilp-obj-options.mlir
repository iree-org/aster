// Tests for ilp-objective option-validation in amdgcn-register-coloring. The
// ILP back-end is always available (no build-time guard), so no ilp_regalloc
// feature is needed.

// RUN: aster-opt --amdgcn-register-coloring="ilp-objective=bad" \
// RUN:   --verify-diagnostics %s

// Verify that an unknown ILP objective name is rejected.
amdgcn.module @bad_objective_mod target = <gfx942> {
  // expected-error @below {{ilp-objective must be "min-pressure" or "feasibility", got "bad"}}
  amdgcn.kernel @bad_objective {
    amdgcn.end_kernel
  }
}
