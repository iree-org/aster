// RUN: aster-opt --amdgcn-register-coloring --verify-diagnostics %s

// Verify that the register allocator rejects a quotient graph node with value
// semantics. The kernel below coalesces B and C (no interference between them)
// and then creates an interference edge between the coalesced class and A (a
// value-semantics alloca). The allocator detects A's value semantics during the
// pre-check in RegisterColoring::run and emits the expected diagnostic.

amdgcn.module @coalescing_value_semantics_neighbor_mod target = <gfx942> {
  // expected-error @below {{failed to run register allocator}}
  amdgcn.kernel @coalescing_value_semantics_neighbor {
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    %c = amdgcn.alloca : !amdgcn.vgpr<?>
    // expected-error @below {{found unexpected value register}}
    %a = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
    lsir.copy %c, %b : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    amdgcn.test_inst ins %a, %c : (!amdgcn.vgpr, !amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}
