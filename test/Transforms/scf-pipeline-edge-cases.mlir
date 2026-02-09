// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// ============================================================================
// No sched.stage attributes: loop should be unchanged.
// ============================================================================

// CHECK-LABEL: func.func @no_stages
// CHECK:         scf.for
// CHECK:           amdgcn.test_inst
// CHECK-NOT:     amdgcn.test_inst
// CHECK:         return
func.func @no_stages() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %v = amdgcn.test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// All ops at stage 0: maxStage=0, no pipelining needed.
// Loop is preserved with sched.stage attributes intact.
// ============================================================================

// CHECK-LABEL: func.func @all_stage_zero
// CHECK:         scf.for
// CHECK:           amdgcn.test_inst
// CHECK:           amdgcn.test_inst
// CHECK-NOT:     amdgcn.test_inst
// CHECK:         return
func.func @all_stage_zero() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Same-stage ops mixed with cross-stage: only the cross-stage value
// becomes an iter_arg, same-stage values resolve within the iteration.
// %a (stage 0) crosses to stage 1 -> iter_arg
// %b (stage 1) used by %c (stage 1) -> same-stage, no iter_arg
//
// Epilogue must freshly clone the same-stage %b for the epilogue section,
// since %b is stage 1 and only exists within the iteration.
// ============================================================================

// CHECK-LABEL: func.func @mixed_same_and_cross_stage

// Prologue: stage 0 at iteration 0
// CHECK:       %[[PRO:.*]] = amdgcn.test_inst outs %[[S0:.*]] :

// Kernel: only %a crosses stages, so 1 iter_arg
// CHECK:       %[[KER:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[PRO]]) -> (!amdgcn.vgpr)
// CHECK:         %[[K_A:.*]] = amdgcn.test_inst outs %[[S0]] :
// Stage 1 ops: %b is freshly computed, %c uses cross-stage %a + same-stage %b
// CHECK:         %[[K_B:.*]] = amdgcn.test_inst outs %[[S1:.*]] :
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[ARG]], %[[K_B]]
// CHECK:         scf.yield %[[K_A]] : !amdgcn.vgpr

// Epilogue: same-stage %b is freshly cloned (not from kernel results)
// CHECK:       %[[E_B:.*]] = amdgcn.test_inst outs %[[S1]] :
// Epilogue consumer uses kernel result for cross-stage %a + fresh %b
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]], %[[E_B]]
// CHECK:       return

func.func @mixed_same_and_cross_stage() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = amdgcn.test_inst outs %s1 {sched.stage = 1 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %c = amdgcn.test_inst outs %s2 ins %a, %b {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
