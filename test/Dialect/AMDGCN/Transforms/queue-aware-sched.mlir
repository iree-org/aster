// RUN: aster-opt %s --aster-apply-sched=scheds=sched,sched_val | FileCheck %s

// Tests for QueueAwareSchedAttr: the same greedy latency-hiding logic as
// amdgcn-low-level-scheduler but expressed as a sched attribute.
//
// Two scheduler compositions:
//   #sched     - SSASchedulerAttr graph builder; used with amdgcn.test_inst ops
//                that lack MemoryEffectOpInterface (ValueSchedulerAttr would
//                treat them as sync points and chain all ops sequentially).
//   #sched_val - ValueSchedulerAttr graph builder; used for tests that need
//                barrier / wait-dependency semantics on real AMDGCN ops.

#sched = #aster_utils.generic_scheduler<
    #amdgcn.ssa_scheduler,
    #aster_utils.sched_stage_labeler,
    #amdgcn.queue_aware_sched>

#sched_val = #aster_utils.generic_scheduler<
    #amdgcn.value_scheduler,
    #aster_utils.sched_stage_labeler,
    #amdgcn.queue_aware_sched>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

// High-latency ops should be scheduled before low-latency fillers.
// Using sched.queue/sched.exec_latency attrs on test_inst to drive classification.
//
// Input: 4 VALU (latency=4) then 2 XDL (latency=16), all independent.
// Expected: the 2 XDL ops first (higher latency), then the 4 VALU ops.
//
// CHECK-LABEL: func.func @xdl_before_valu
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "xdl"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "xdl"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         return
func.func @xdl_before_valu() attributes {sched = #sched} {
  %v0 = amdgcn.alloca : !v
  %v1 = amdgcn.alloca : !v
  %v2 = amdgcn.alloca : !v
  %v3 = amdgcn.alloca : !v
  %v4 = amdgcn.alloca : !v
  %v5 = amdgcn.alloca : !v
  // input: valu, valu, xdl, valu, valu, xdl (all independent)
  %f0 = amdgcn.test_inst outs %v0 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %f1 = amdgcn.test_inst outs %v1 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %m0 = amdgcn.test_inst outs %v2 {sched.queue = "xdl", sched.exec_latency = 16 : i64} : (!v) -> !v
  %f2 = amdgcn.test_inst outs %v3 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %f3 = amdgcn.test_inst outs %v4 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %m1 = amdgcn.test_inst outs %v5 {sched.queue = "xdl", sched.exec_latency = 16 : i64} : (!v) -> !v
  return
}

// LGKM (LDS) reads before VALU fillers.
//
// Input: 4 VALU (latency=4) then 2 LGKM (latency=32), all independent.
// Expected: the 2 LGKM ops first, then the 4 VALU ops.
//
// CHECK-LABEL: func.func @lgkm_before_valu
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "lgkm"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "lgkm"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         return
func.func @lgkm_before_valu() attributes {sched = #sched} {
  %v0 = amdgcn.alloca : !v
  %v1 = amdgcn.alloca : !v
  %v2 = amdgcn.alloca : !v
  %v3 = amdgcn.alloca : !v
  %v4 = amdgcn.alloca : !v
  %v5 = amdgcn.alloca : !v
  // input: valu, valu, lgkm, valu, valu, lgkm (all independent)
  %f0 = amdgcn.test_inst outs %v0 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %f1 = amdgcn.test_inst outs %v1 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %d0 = amdgcn.test_inst outs %v2 {sched.queue = "lgkm", sched.exec_latency = 32 : i64} : (!v) -> !v
  %f2 = amdgcn.test_inst outs %v3 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %f3 = amdgcn.test_inst outs %v4 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %d1 = amdgcn.test_inst outs %v5 {sched.queue = "lgkm", sched.exec_latency = 32 : i64} : (!v) -> !v
  return
}

// VMEM queue depth is 2; the third VMEM op stalls and gets interleaved with VALU.
//
// Input: valu, vmem, valu, vmem, valu, vmem, valu (alternating).
// Expected: the first 2 VMEM burst together, then 4 VALU (to hide stall), then the 3rd VMEM.
//
// CHECK-LABEL: func.func @vmem_interleave_valu
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "vmem"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "vmem"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "vmem"
// CHECK:         return
func.func @vmem_interleave_valu() attributes {sched = #sched} {
  %v0 = amdgcn.alloca : !v
  %v1 = amdgcn.alloca : !v
  %v2 = amdgcn.alloca : !v
  %v3 = amdgcn.alloca : !v
  %v4 = amdgcn.alloca : !v
  %v5 = amdgcn.alloca : !v
  %v6 = amdgcn.alloca : !v
  // input: valu, vmem, valu, vmem, valu, vmem, valu (bad order)
  %r0 = amdgcn.test_inst outs %v0 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %t0 = amdgcn.test_inst outs %v1 {sched.queue = "vmem", sched.exec_latency = 128 : i64} : (!v) -> !v
  %r1 = amdgcn.test_inst outs %v2 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %t1 = amdgcn.test_inst outs %v3 {sched.queue = "vmem", sched.exec_latency = 128 : i64} : (!v) -> !v
  %r2 = amdgcn.test_inst outs %v4 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %t2 = amdgcn.test_inst outs %v5 {sched.queue = "vmem", sched.exec_latency = 128 : i64} : (!v) -> !v
  %r3 = amdgcn.test_inst outs %v6 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  return
}

// SSA deps must be respected: addr must come before its dependent load.
//
// CHECK-LABEL: func.func @respects_ssa_deps
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "valu"{{.*}}-> !amdgcn.vgpr
// CHECK:         amdgcn.test_inst {{.*}}sched.queue = "vmem"
// CHECK:         return
func.func @respects_ssa_deps() attributes {sched = #sched} {
  %v0 = amdgcn.alloca : !v
  %v1 = amdgcn.alloca : !v
  // vmem op depends on addr (valu); addr must come first despite lower latency
  %addr = amdgcn.test_inst outs %v0 {sched.queue = "valu", sched.exec_latency = 4 : i64} : (!v) -> !v
  %load = amdgcn.test_inst outs %v1 ins %addr {sched.queue = "vmem", sched.exec_latency = 128 : i64} : (!v, !v) -> !v
  return
}

// s_waitcnt is a barrier: VALU ops must not move past it.
//
// CHECK-LABEL: func.func @waitcnt_is_barrier
// CHECK:         vop2 v_add_u32
// CHECK:         sopp.s_waitcnt
// CHECK:         vop2 v_add_u32
// CHECK:         return
func.func @waitcnt_is_barrier(%v0: !v, %v1: !v, %v2: !v, %v3: !v)
    attributes {sched = #sched_val} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // valu before waitcnt
  %r0 = amdgcn.vop2 v_add_u32 outs %v0 ins %c0, %v1 : !v, i32, !v
  amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
  // valu after waitcnt -- must stay after
  %r1 = amdgcn.vop2 v_add_u32 outs %v2 ins %c1, %v3 : !v, i32, !v
  return
}

// Independent VALU ops schedule across a wait.
//
// CHECK-LABEL: func.func @valu_across_wait
// CHECK:         load ds_read_b64
// wait stays before its data consumer (the MFMA)
// CHECK:         wait deps
// CHECK:         vop3p_mai
// independent VALU scheduled past the wait (no dep on waited data)
// CHECK:         vop2 v_add_u32
// CHECK:         return
func.func @valu_across_wait(%va: !v, %vb: !v, %vc: !v, %lds_addr: !v,
                             %lds_d: !vx2, %a: !vx2, %b: !vx2,
                             %acc0: !vx4, %dst: !vx4)
    attributes {sched = #sched_val} {
  %c0 = arith.constant 0 : i32
  %c42 = arith.constant 42 : i32
  // lds load produces data + token
  %lds_data, %tok = amdgcn.load ds_read_b64 dest %lds_d addr %lds_addr offset c(%c0)
      : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
  // independent valu (no dep on lds data)
  %r0 = amdgcn.vop2 v_add_u32 outs %va ins %c42, %vb : !v, i32, !v
  // wait for lds
  amdgcn.wait deps %tok : !amdgcn.read_token<shared>
  // mfma uses lds data (must come after wait)
  %mfma = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %lds_data, %b, %acc0
      : !vx2, !vx2, !vx4 -> !vx4
  return
}
