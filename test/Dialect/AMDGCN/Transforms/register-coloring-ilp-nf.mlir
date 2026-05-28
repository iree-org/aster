// REQUIRES: ilp_regalloc
// RUN: aster-opt --amdgcn-register-coloring="reg-alloc-solver=ilp num-vgprs=2" \
// RUN:   --verify-diagnostics %s
// RUN: aster-opt \
// RUN:   --amdgcn-register-coloring="reg-alloc-solver=ilp ilp-objective=feasibility num-vgprs=2" \
// RUN:   --verify-diagnostics %s

// Verify that the ILP allocator returns failure when three mutually-live VGPRs
// cannot fit into two slots — for both the min-pressure and feasibility
// objectives — matching the greedy allocator's behaviour.

amdgcn.module @ilp_vgpr_exhaustion_mod target = <gfx942> {
  // expected-error @below {{failed to run register allocator}}
  amdgcn.kernel @ilp_vgpr_exhaustion {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    %c = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst outs %c : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst ins %a, %b, %c : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}
