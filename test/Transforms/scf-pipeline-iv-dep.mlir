// RUN: aster-opt --aster-scf-pipeline %s | FileCheck %s

// ============================================================================
// Two-stage pipeline where stage 1 uses the induction variable.
// The kernel must adjust the IV for stage 1: kernelIV - 1*step.
//
// Original: for i in [0, 4): produce(i), index_cast(i) + consume(produce(i))
// Pipelined:
//   Prologue: produce(0)
//   Kernel:   for ki in [1, 4):
//               produce(ki), index_cast(ki - 1), consume(produce(ki-1))
//   Epilogue: index_cast(3), consume(produce(3))
// ============================================================================

// CHECK-LABEL: func.func @iv_in_stage1

// Prologue: stage 0 at iter 0 (index_cast is stage 1, not emitted here)
// CHECK:       %[[PRO_V:.*]] = amdgcn.test_inst

// Kernel: cross-stage value as iter_arg
// CHECK:       scf.for %[[KI:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[PRO_V]]) -> (!amdgcn.vgpr)

// Stage 0: uses kernelIV directly
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %{{.*}}

// Stage 1: adjusted IV = kernelIV - step
// CHECK:         %[[ADJ:.*]] = arith.subi %[[KI]],
// CHECK:         arith.index_cast %[[ADJ]] : index to i32
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[ARG]]
// CHECK:         scf.yield %[[K_V]] : !amdgcn.vgpr

// Epilogue: stage 1 at iter 3
// CHECK:       arith.index_cast
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER:.*]]
// CHECK:       return

func.func @iv_in_stage1() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %idx = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Two-stage pipeline where BOTH stages use the IV.
// Stage 0 uses IV at iteration i, stage 1 uses IV at iteration i-1.
// The kernel should use kernelIV for stage 0 and kernelIV - step for stage 1.
// ============================================================================

// CHECK-LABEL: func.func @iv_in_both_stages

// Prologue: stage 0, iter 0 -- index_cast uses IV=0
// CHECK:       arith.index_cast %{{.*}} : index to i32
// CHECK:       %[[PRO_V:.*]] = amdgcn.test_inst

// Kernel
// CHECK:       scf.for %[[KI:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[PRO_V]]) -> (!amdgcn.vgpr)

// Stage 0: index_cast uses kernelIV directly
// CHECK:         arith.index_cast %[[KI]] : index to i32
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %{{.*}}

// Stage 1: adjusted IV = kernelIV - step
// CHECK:         %[[ADJ:.*]] = arith.subi %[[KI]],
// CHECK:         arith.index_cast %[[ADJ]] : index to i32
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[ARG]]
// CHECK:         scf.yield %[[K_V]] : !amdgcn.vgpr

// Epilogue: stage 1 at iter 3
// CHECK:       arith.index_cast
// CHECK:       amdgcn.test_inst
// CHECK:       return

func.func @iv_in_both_stages() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %idx0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %idx1 = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Three-stage pipeline where all stages use the IV.
// Stage 0: index_cast at iter i
// Stage 1: index_cast at iter i-1 + consume cross-stage value
// Stage 2: index_cast at iter i-2
//
// Kernel should produce three different adjusted IVs:
//   stage 0: kernelIV
//   stage 1: kernelIV - step
//   stage 2: kernelIV - 2*step
// ============================================================================

// CHECK-LABEL: func.func @iv_three_stages

// Prologue section 0: stage 0 at iter 0
// CHECK:       arith.index_cast
// CHECK:       %{{.*}} = amdgcn.test_inst

// Prologue section 1: stage 0 at iter 1, stage 1 at iter 0
// CHECK:       arith.index_cast
// CHECK:       %[[P1_V:.*]] = amdgcn.test_inst
// CHECK:       arith.index_cast
// CHECK:       %[[P1_W:.*]] = amdgcn.test_inst

// Kernel: 2 iter_args (from stage 0->1 and stage 1->2)
// CHECK:       %[[KER:.*]]:2 = scf.for %[[KI:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A0:.*]] = %[[P1_V]], %[[A1:.*]] = %[[P1_W]]) -> (!amdgcn.vgpr, !amdgcn.vgpr)

// Stage 0: uses kernelIV
// CHECK:         arith.index_cast %[[KI]] : index to i32
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst

// Stage 1: uses kernelIV - step
// CHECK:         %[[ADJ1:.*]] = arith.subi %[[KI]],
// CHECK:         arith.index_cast %[[ADJ1]] : index to i32
// CHECK:         %[[K_W:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[A0]]

// Stage 2: uses kernelIV - 2*step
// CHECK:         %[[ADJ2:.*]] = arith.subi %[[KI]],
// CHECK:         arith.index_cast %[[ADJ2]] : index to i32
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[A1]]
// CHECK:         scf.yield %[[K_V]], %[[K_W]] : !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue
// CHECK:       return

func.func @iv_three_stages() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %idx0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %idx1 = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %idx2 = arith.index_cast %i {sched.stage = 2 : i32} : index to i32
    %x = amdgcn.test_inst outs %s2 ins %w {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// ============================================================================
// Two-stage pipeline with step > 1.
// Step = 2 means the IV adjustment for stage 1 is kernelIV - 2 (not -1).
// ============================================================================

// CHECK-LABEL: func.func @iv_step_two

// Prologue: stage 0 at iter 0
// CHECK:       arith.index_cast
// CHECK:       %[[PRO_V:.*]] = amdgcn.test_inst

// Kernel
// CHECK:       scf.for %[[KI:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args

// Stage 0: uses kernelIV directly
// CHECK:         arith.index_cast %[[KI]] : index to i32

// Stage 1: adjusted IV = kernelIV - 2 (step=2, stage=1, so offset = 1*2 = 2)
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[ADJ:.*]] = arith.subi %[[KI]], %[[C2]]
// CHECK:         arith.index_cast %[[ADJ]] : index to i32
// CHECK:         scf.yield

// Epilogue
// CHECK:       return

func.func @iv_step_two() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c2 {
    %idx0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %idx1 = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
