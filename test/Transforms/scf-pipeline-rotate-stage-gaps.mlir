// RUN: aster-opt --aster-scf-pipeline="enable-rotate-stage=true rotate-stage=2" %s -split-input-file | FileCheck %s

// `test_tag` matches `sched.stage`. Pipeliner strips `sched.stage`, so only
// `test_tag` survives. Captures thread the SSA chain through prologue, kernel,
// and epilogue.

// CHECK-LABEL: func.func @rotate_with_stage_gap
// CHECK-NOT:   sched.stage
// CHECK:       %[[GL0:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[GL1:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[RD0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL0]] {test_tag = 2 : i32}
// CHECK:       %[[GL2:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[RD1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL1]] {test_tag = 2 : i32}
// CHECK:       %[[MFMA0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[RD0]] {test_tag = 3 : i32}
// CHECK:       %[[GL3:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[KER:.*]]:4 = scf.for {{.*}} iter_args(%[[GL_NEW:.*]] = %[[GL3]], %[[GL_OLD:.*]] = %[[GL2]], %[[RD_PREV:.*]] = %[[RD1]], %[[MFMA_PREV:.*]] = %[[MFMA0]]) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[ROT:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL_OLD]] {test_tag = 2 : i32}
// CHECK:         %[[MFMA:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[RD_PREV]] {test_tag = 3 : i32}
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[MFMA_PREV]] {test_tag = 4 : i32}
// CHECK:         %[[GL:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:         scf.yield %[[GL]], %[[GL_NEW]], %[[ROT]], %[[MFMA]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:       %[[E_RD0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1 {test_tag = 2 : i32}
// CHECK:       %[[E_MFMA0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#2 {test_tag = 3 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#3 {test_tag = 4 : i32}
// CHECK:       %[[E_RD1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#0 {test_tag = 2 : i32}
// CHECK:       %[[E_MFMA1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[E_RD0]] {test_tag = 3 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[E_MFMA0]] {test_tag = 4 : i32}
// CHECK:       %[[E_MFMA2:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[E_RD1]] {test_tag = 3 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[E_MFMA1]] {test_tag = 4 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[E_MFMA2]] {test_tag = 4 : i32}

func.func @rotate_with_stage_gap() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %s3 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %gl = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32, test_tag = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %rd = amdgcn.test_inst outs %s1 ins %gl {sched.stage = 2 : i32, test_tag = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %mfma = amdgcn.test_inst outs %s2 ins %rd {sched.stage = 3 : i32, test_tag = 3 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %nxt = amdgcn.test_inst outs %s3 ins %mfma {sched.stage = 4 : i32, test_tag = 4 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----

// CHECK-LABEL: func.func @rotate_with_trailing_gap
// CHECK-NOT:   sched.stage
// CHECK:       %[[GL0:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[GL1:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[WR0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL0]] {test_tag = 1 : i32}
// CHECK:       %[[RD0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[WR0]] {test_tag = 2 : i32}
// CHECK:       %[[GL2:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[WR1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL1]] {test_tag = 1 : i32}
// CHECK:       %[[RD1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[WR1]] {test_tag = 2 : i32}
// CHECK:       %[[GL3:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       %[[WR2:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL2]] {test_tag = 1 : i32}
// CHECK:       %[[KER:.*]]:4 = scf.for {{.*}} iter_args(%[[WR_PREV:.*]] = %[[WR2]], %[[RD_NEW:.*]] = %[[RD1]], %[[RD_OLD:.*]] = %[[RD0]], %[[GL_PREV:.*]] = %[[GL3]]) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:         %[[RD:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[WR_PREV]] {test_tag = 2 : i32}
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[RD_OLD]] {test_tag = 4 : i32}
// CHECK:         %[[GL:.*]] = amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:         %[[WR:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[GL_PREV]] {test_tag = 1 : i32}
// CHECK:         scf.yield %[[WR]], %[[RD]], %[[RD_NEW]], %[[GL]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:       %[[E_RD0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#0 {test_tag = 2 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#2 {test_tag = 4 : i32}
// CHECK:       %[[E_WR0:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[KER]]#3 {test_tag = 1 : i32}
// CHECK:       %[[E_RD1:.*]] = amdgcn.test_inst outs %{{.*}} ins %[[E_WR0]] {test_tag = 2 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]#1 {test_tag = 4 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[E_RD0]] {test_tag = 4 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[E_RD1]] {test_tag = 4 : i32}

func.func @rotate_with_trailing_gap() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %s3 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  scf.for %i = %c0 to %c7 step %c1 {
    %gl = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32, test_tag = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %wr = amdgcn.test_inst outs %s1 ins %gl {sched.stage = 1 : i32, test_tag = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %rd = amdgcn.test_inst outs %s2 ins %wr {sched.stage = 2 : i32, test_tag = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %nxt = amdgcn.test_inst outs %s3 ins %rd {sched.stage = 4 : i32, test_tag = 4 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
