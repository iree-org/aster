// RUN: aster-opt %s --aster-apply-sched=scheds=sched --allow-unregistered-dialect | FileCheck %s

#sched = #aster_utils.generic_scheduler<#amdgcn.register_scheduler, #aster_utils.sched_stage_labeler, #aster_utils.stage_topo_sort_sched>

// Post-register-semantics DPS kernel (`!amdgcn.vgpr<?>`). Register scheduler adds
// reaching-definition edges; schedule must keep the vop2 chain order.
// CHECK-LABEL:   func.func @ssa_chain_register(
// CHECK-SAME:  ) {
// CHECK:           %[[A0:.*]] = amdgcn.alloca
// CHECK:           %[[A1:.*]] = amdgcn.alloca
// CHECK:           %[[A2:.*]] = amdgcn.alloca
// CHECK:           %[[A3:.*]] = amdgcn.alloca
// CHECK:           amdgcn.vop2 v_add_u32 outs %[[A3]] ins %[[A1]], %[[A0]] {sched.stage = 1 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           amdgcn.vop2 v_add_u32 outs %[[A2]] ins %[[A1]], %[[A0]] {sched.stage = 2 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           amdgcn.vop2 v_add_u32 outs %[[A2]] ins %[[A2]], %[[A3]] {sched.stage = 0 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           return
// CHECK:         }
func.func @ssa_chain_register() attributes {sched = #sched} {
  %0 = amdgcn.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca {sched.stage = 0 : i32} : !amdgcn.vgpr<?>
  amdgcn.vop2 v_add_u32 outs %2 ins %1, %0 {sched.stage = 2 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.vop2 v_add_u32 outs %3 ins %1, %0 {sched.stage = 1 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.vop2 v_add_u32 outs %2 ins %2, %3 {sched.stage = 0 : i32} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}
