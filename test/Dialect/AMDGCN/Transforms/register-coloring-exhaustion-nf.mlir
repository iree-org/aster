// RUN: aster-opt --amdgcn-register-coloring="num-vgprs=2" --verify-diagnostics %s

// Verify that the register allocator rejects a kernel whose VGPR demand
// exceeds the configured limit. Three mutually-live VGPRs cannot fit in two
// slots; each alloca is written before the final consumer makes all three live
// simultaneously.

// expected-error @below {{failed to allocate the registers}}
// expected-error @below {{failed to run register allocator}}
amdgcn.kernel @vgpr_exhaustion {
  %a = amdgcn.alloca : !amdgcn.vgpr<?>
  %b = amdgcn.alloca : !amdgcn.vgpr<?>
  %c = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %a : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst outs %c : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst ins %a, %b, %c : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  amdgcn.end_kernel
}
