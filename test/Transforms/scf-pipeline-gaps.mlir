// RUN: aster-opt --aster-scf-pipeline --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @gap_0_2

// Prologue: 2 sections, both only stage 0
// CHECK:       %[[PRO0:.*]] = amdgcn.test_inst outs %[[S0:.*]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       %[[PRO1:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr

// Kernel: shift register of depth 2 -- iter_args(%arg1=newest, %arg2=oldest)
// CHECK:       %[[KER:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %[[PRO1]], %[[A2:.*]] = %[[PRO0]]) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// Stage 2 consumes oldest slot (arg2)
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[A2]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// Yield: fresh value, then shift
// CHECK:         scf.yield %[[K_V]], %[[A1]] : !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue: 2 sections, both only stage 2
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       return

func.func @gap_0_2() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----


// CHECK-LABEL: func.func @gap_0_2_iv

// Prologue: stage 0 ops at iter 0 and iter 1 (index_cast + test_inst each)
// CHECK:       arith.index_cast
// CHECK:       %[[PRO0:.*]] = amdgcn.test_inst outs %[[S0:.*]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       arith.index_cast
// CHECK:       %[[PRO1:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr

// Kernel: shift register depth 2, plus IV adjustment (subi by 2)
// CHECK:       %[[KER:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %[[PRO1]], %[[A2:.*]] = %[[PRO0]]) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// Stage 0: index_cast + produce
// CHECK:         arith.index_cast
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// Stage 2: IV adjusted by -2, index_cast + consume oldest
// CHECK:         affine.apply
// CHECK:         arith.index_cast
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[A2]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:         scf.yield %[[K_V]], %[[A1]] : !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue: 2 sections with IV-adjusted stage-2 ops
// CHECK:       arith.index_cast
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       arith.index_cast
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       return

func.func @gap_0_2_iv() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %idx0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %idx2 = arith.index_cast %i {sched.stage = 2 : i32} : index to i32
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----


// CHECK-LABEL: func.func @gap_0_2_iter_args

// Prologue section 0: stage 0 at iter 0 (to_reg(init0), produce)
// CHECK-DAG:   %[[INIT0:.*]] = arith.constant 10 : i32
// CHECK-DAG:   %[[INIT1:.*]] = arith.constant 20 : i32
// CHECK:       lsir.to_reg %[[INIT0]] :
// CHECK:       %[[PRO0:.*]] = amdgcn.test_inst outs %[[S0:.*]] ins
// Prologue section 1: stage 0 at iter 1 (to_reg(init1 after swap), produce)
// After yield sim for section 0: (a,b)=(init0,init1) -> swap -> (init1,init0)
// So section 1 uses init1 (which is now a after swap)
// CHECK:       lsir.to_reg %[[INIT1]] :
// CHECK:       %[[PRO1:.*]] = amdgcn.test_inst outs %[[S0]] ins

// Kernel: shift register (2 csv slots) + 2 user iter_args = 4 iter_args
// CHECK:       %[[KER:.*]]:4 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %[[PRO1]], %[[A2:.*]] = %[[PRO0]], %[[UA:.*]] = %{{.*}}, %[[UB:.*]] = %{{.*}}) -> (!amdgcn.vgpr, !amdgcn.vgpr, i32, i32)
// Stage 0: to_reg(user iter_arg a) + produce
// CHECK:         lsir.to_reg %[[UA]] :
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]]
// Stage 2: consume oldest csv slot
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[A2]] :
// Yield: csv shift + user iter_arg swap
// CHECK:         scf.yield %[[K_V]], %[[A1]], %[[UB]], %[[UA]] : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32

// Epilogue: 2 sections, stage 2 only
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#1 :
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 :
// CHECK:       return

func.func @gap_0_2_iter_args() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %init0 = arith.constant 10 : i32
  %init1 = arith.constant 20 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %r:2 = scf.for %i = %c0 to %c6 step %c1
      iter_args(%a = %init0, %b = %init1) -> (i32, i32) {
    %addr = lsir.to_reg %a {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
    %v = amdgcn.test_inst outs %s0 ins %addr {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %b, %a : i32, i32
  }
  return
}

// -----


// CHECK-LABEL: func.func @gap_0_2_5

// Prologue sections 0-1: stage 0 only
// CHECK:       %[[P0_V:.*]] = amdgcn.test_inst outs %[[S0:.*]] :
// CHECK:       %[[P1_V:.*]] = amdgcn.test_inst outs %[[S0]] :
// Prologue section 2: stage 0 + stage 2 (stage 2 consumes iter 0 produce)
// CHECK:       %[[P2_V:.*]] = amdgcn.test_inst outs %[[S0]] :
// CHECK:       %[[P2_W:.*]] = amdgcn.test_inst outs %[[S1:.*]] ins %[[P0_V]] :
// Prologue section 3: stage 0 + stage 2
// CHECK:       %[[P3_V:.*]] = amdgcn.test_inst outs %[[S0]] :
// CHECK:       %[[P3_W:.*]] = amdgcn.test_inst outs %[[S1]] ins %[[P1_V]] :
// Prologue section 4: stage 0 + stage 2
// CHECK:       %[[P4_V:.*]] = amdgcn.test_inst outs %[[S0]] :
// CHECK:       %[[P4_W:.*]] = amdgcn.test_inst outs %[[S1]] ins %[[P2_V]] :

// Kernel: 5 iter_args -- csv0 (depth 2): [newest, oldest], csv1 (depth 3): [newest, mid, oldest]
// CHECK:       %[[KER:.*]]:5 = scf.for {{.*}} iter_args(%[[CA1:.*]] = %{{.*}}, %[[CA2:.*]] = %{{.*}}, %[[CB1:.*]] = %{{.*}}, %[[CB2:.*]] = %{{.*}}, %[[CB3:.*]] = %{{.*}}) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr)
// Stage 0: produce
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]] :
// Stage 2: consume oldest csv0 slot, produce W
// CHECK:         %[[K_W:.*]] = amdgcn.test_inst outs %[[S1]] ins %[[CA2]] :
// Stage 5: consume oldest csv1 slot
// CHECK:         amdgcn.test_inst outs %[[S2:.*]] ins %[[CB3]] :
// Yield: csv0 shift (fresh, shift), csv1 shift (fresh W, shift, shift)
// CHECK:         scf.yield %[[K_V]], %[[CA1]], %[[K_W]], %[[CB1]], %[[CB2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue: stages 2 and 5 drain
// Epilogue section 1: stage 2 + stage 5
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#1 :
// CHECK:       amdgcn.test_inst outs %[[S2]] ins %[[KER]]#4 :
// Epilogue section 2: stage 2 + stage 5
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 :
// CHECK:       amdgcn.test_inst outs %[[S2]] ins %[[KER]]#3 :
// Epilogue sections 3-5: stage 5 only (consuming from previous stage 2 results)
// CHECK:       amdgcn.test_inst outs %[[S2]]
// CHECK:       amdgcn.test_inst outs %[[S2]]
// CHECK:       amdgcn.test_inst outs %[[S2]]
// CHECK:       return

func.func @gap_0_2_5() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %x = amdgcn.test_inst outs %s2 ins %w {sched.stage = 5 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----


// CHECK-LABEL: func.func @gap_0_3

// Prologue: 3 sections, all only stage 0
// CHECK:       %[[PRO0:.*]] = amdgcn.test_inst outs %[[S0:.*]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       %[[PRO1:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       %[[PRO2:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr

// Kernel: shift register of depth 3 -- iter_args(newest, middle, oldest)
// CHECK:       %[[KER:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %[[PRO2]], %[[A2:.*]] = %[[PRO1]], %[[A3:.*]] = %[[PRO0]]) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[K_V:.*]] = amdgcn.test_inst outs %[[S0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// Stage 3 consumes oldest slot
// CHECK:         amdgcn.test_inst outs %[[S1:.*]] ins %[[A3]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// Yield: fresh, shift forward
// CHECK:         scf.yield %[[K_V]], %[[A1]], %[[A2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

// Epilogue: 3 sections, all only stage 3
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#2 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       amdgcn.test_inst outs %[[S1]] ins %[[KER]]#0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:       return

func.func @gap_0_3() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 3 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
