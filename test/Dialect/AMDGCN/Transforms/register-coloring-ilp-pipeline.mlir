// REQUIRES: ilp_regalloc
// RUN: aster-opt %s --amdgcn-reg-alloc="reg-alloc-solver=ilp" \
// RUN:   --split-input-file | FileCheck %s

// Verify that the reg-alloc-solver=ilp option propagates correctly through the
// amdgcn-reg-alloc pipeline into amdgcn-register-coloring. Two non-interfering
// VGPRs should collapse to a single slot under the min-pressure objective.

// CHECK-LABEL: kernel @pipeline_ilp_propagation {
// CHECK-DAG:     %[[V:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-NOT:     !amdgcn.vgpr<1>
// CHECK:         end_kernel
amdgcn.module @pipeline_ilp_propagation_mod target = <gfx942> {
  amdgcn.kernel @pipeline_ilp_propagation {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}
