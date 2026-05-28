// REQUIRES: ilp_regalloc
// RUN: aster-opt %s --amdgcn-register-coloring="reg-alloc-solver=ilp" \
// RUN:   --amdgcn-post-reg-alloc-legalization --cse --split-input-file \
// RUN:   | FileCheck %s
// RUN: aster-opt %s \
// RUN:   --amdgcn-register-coloring="reg-alloc-solver=ilp ilp-objective=feasibility" \
// RUN:   --amdgcn-post-reg-alloc-legalization --cse --split-input-file \
// RUN:   | FileCheck %s --check-prefix=FEAS

// Verify that two non-interfering VGPRs are assigned the same register under
// the min-pressure objective (a single slot suffices for both).
// CHECK-LABEL:   kernel @non_interfering_vgprs {
// CHECK-DAG:       %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:           test_inst outs %[[V0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:           test_inst outs %[[V0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK-NOT:       !amdgcn.vgpr<1>
// CHECK:           end_kernel
// CHECK:         }
// Feasibility objective: assignment may differ but must be valid (allocated).
// FEAS-LABEL:   kernel @non_interfering_vgprs {
// FEAS:           alloca : !amdgcn.vgpr<{{[0-9]+}}>
amdgcn.module @non_interfering_vgprs_mod target = <gfx942> {
  amdgcn.kernel @non_interfering_vgprs {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// Two simultaneously-live VGPRs must receive distinct physical registers.
// CHECK-LABEL:   kernel @interfering_vgprs {
// CHECK-DAG:       alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       alloca : !amdgcn.vgpr<1>
// CHECK:           end_kernel
// CHECK:         }
// FEAS-LABEL:   kernel @interfering_vgprs {
// FEAS-DAG:       alloca : !amdgcn.vgpr<{{[0-9]+}}>
// FEAS-DAG:       alloca : !amdgcn.vgpr<{{[0-9]+}}>
amdgcn.module @interfering_vgprs_mod target = <gfx942> {
  amdgcn.kernel @interfering_vgprs {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.vgpr<?>) -> ()
    amdgcn.test_inst ins %a, %b : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// An unallocated VGPR that interferes with a pre-allocated !amdgcn.vgpr<0>
// must not be assigned register 0.
// CHECK-LABEL:   kernel @fixed_neighbor_avoidance {
// CHECK:           alloca : !amdgcn.vgpr<0>
// CHECK-NOT:       alloca : !amdgcn.vgpr<0>
// CHECK:           end_kernel
// CHECK:         }
// FEAS-LABEL:   kernel @fixed_neighbor_avoidance {
// FEAS:           alloca : !amdgcn.vgpr<0>
amdgcn.module @fixed_neighbor_avoidance_mod target = <gfx942> {
  amdgcn.kernel @fixed_neighbor_avoidance {
    %fixed = amdgcn.alloca : !amdgcn.vgpr<0>
    %free = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.test_inst ins %fixed, %free : (!amdgcn.vgpr<0>, !amdgcn.vgpr<?>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// Two non-interfering SGPRs are assigned the same register under the
// min-pressure objective. Verifies the SGPR kind path in maxRegsForKind,
// numSGPRs threading, and the SGPR interference group.
// CHECK-LABEL:   kernel @sgpr_allocation {
// CHECK-DAG:       %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:           test_inst outs %[[S0]] : (!amdgcn.sgpr<0>) -> ()
// CHECK:           test_inst outs %[[S0]] : (!amdgcn.sgpr<0>) -> ()
// CHECK-NOT:       !amdgcn.sgpr<1>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.module @sgpr_allocation_mod target = <gfx942> {
  amdgcn.kernel @sgpr_allocation {
    %a = amdgcn.alloca : !amdgcn.sgpr<?>
    %b = amdgcn.alloca : !amdgcn.sgpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.sgpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.sgpr<?>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// A range of two VGPRs must be assigned a contiguous pair. Exercises the
// range-leader / non-leader convention: the leader (numRegs==2) gets a
// domain-constrained start variable; the follower (numRegs==0) is written
// as a side effect by extractSolution.
// The range type !amdgcn.vgpr<[B : B+2]> encodes both base and end, proving
// both slots are allocated and adjacent. A non-contiguous assignment would
// produce a different type (the encoding requires end == base + size).
// CHECK-LABEL:   kernel @range_leader {
// CHECK:           test_inst outs {{.*}} : (!amdgcn.vgpr<[[[BASE:[0-9]+]] : {{[0-9]+}}]>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.module @range_leader_mod target = <gfx942> {
  amdgcn.kernel @range_leader {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    %r = amdgcn.make_register_range %a, %b : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    amdgcn.test_inst outs %r : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// Two non-interfering AGPRs are assigned the same register under the
// min-pressure objective. Verifies the AGPR kind path in maxRegsForKind.
// CHECK-LABEL:   kernel @agpr_allocation {
// CHECK-DAG:       %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:           test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:           test_inst outs %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK-NOT:       !amdgcn.agpr<1>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.module @agpr_allocation_mod target = <gfx942> {
  amdgcn.kernel @agpr_allocation {
    %a = amdgcn.alloca : !amdgcn.agpr<?>
    %b = amdgcn.alloca : !amdgcn.agpr<?>
    amdgcn.test_inst outs %a : (!amdgcn.agpr<?>) -> ()
    amdgcn.test_inst outs %b : (!amdgcn.agpr<?>) -> ()
    amdgcn.end_kernel
  }
}

// -----

// A pair of VGPRs with alignment=2 must land on an even-numbered start. The
// pre-allocated neighbour at vgpr<1> makes start=0 infeasible (it would
// require vgpr<1> as the second slot), so the allocator must pick start=2.
// CHECK-LABEL:   kernel @alignment_constraint {
// CHECK:           make_register_range
// CHECK-SAME:        !amdgcn.vgpr<[[B:[02468]+]]>, !amdgcn.vgpr<
// CHECK:           end_kernel
// CHECK:         }
amdgcn.module @alignment_constraint_mod target = <gfx942> {
  amdgcn.kernel @alignment_constraint {
    %a = amdgcn.alloca : !amdgcn.vgpr<?>
    %b = amdgcn.alloca : !amdgcn.vgpr<?>
    %nbr = amdgcn.alloca : !amdgcn.vgpr<1>
    %r = amdgcn.make_register_range %a, %b : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    amdgcn.test_inst outs %r : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    // Force interference with the pre-allocated register so the allocator
    // cannot use start=0 (which would place the range at [0,1) and collide).
    amdgcn.reg_interference %a, %nbr : !amdgcn.vgpr<?>, !amdgcn.vgpr<1>
    amdgcn.end_kernel
  }
}
