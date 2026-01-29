// RUN: aster-opt %s --amdgcn-convert-scf-control-flow --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @test_uniform_loops_const_bounds() {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.scc
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_1]] ins %[[CONSTANT_0]] : !amdgcn.sgpr, i32
// CHECK:           amdgcn.cmpi s_cmp_lt_i32 outs %[[ALLOCA_0]] ins %[[VAL_0]], %[[CONSTANT_2]] : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
// CHECK:           amdgcn.cbranch s_cbranch_scc0 %[[ALLOCA_0]] ^bb2 fallthrough(^bb1) : !amdgcn.scc
// CHECK:         ^bb1:
// CHECK:           %[[FROM_REG_0:.*]] = lsir.from_reg %[[VAL_0]] : !amdgcn.sgpr -> i32
// CHECK:           %[[TO_REG_0:.*]] = lsir.to_reg %[[FROM_REG_0]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG_0]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_add_i32 outs %[[VAL_0]] ins %[[VAL_0]], %[[CONSTANT_1]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           amdgcn.cmpi s_cmp_lt_i32 outs %[[ALLOCA_0]] ins %[[VAL_1]], %[[CONSTANT_2]] : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
// CHECK:           amdgcn.cbranch s_cbranch_scc1 %[[ALLOCA_0]] ^bb1 fallthrough(^bb2) : !amdgcn.scc
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_loops_const_bounds() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  scf.for %i = %c0 to %c10 step %c1 : i32 {
    %iv = lsir.to_reg %i : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @test_uniform_loops_non_const_bounds(
// CHECK-SAME:      %[[ARG0:.*]]: i32) {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG0]] : i32
// CHECK:           %[[TO_REG_0:.*]] = lsir.to_reg %[[ASSUME_UNIFORM_0]] : i32 -> !amdgcn.sgpr
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.scc
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_1]] ins %[[CONSTANT_0]] : !amdgcn.sgpr, i32
// CHECK:           amdgcn.cmpi s_cmp_lt_i32 outs %[[ALLOCA_0]] ins %[[VAL_0]], %[[TO_REG_0]] : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
// CHECK:           amdgcn.cbranch s_cbranch_scc0 %[[ALLOCA_0]] ^bb2 fallthrough(^bb1) : !amdgcn.scc
// CHECK:         ^bb1:
// CHECK:           %[[FROM_REG_0:.*]] = lsir.from_reg %[[VAL_0]] : !amdgcn.sgpr -> i32
// CHECK:           %[[TO_REG_1:.*]] = lsir.to_reg %[[FROM_REG_0]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG_1]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_add_i32 outs %[[VAL_0]] ins %[[VAL_0]], %[[CONSTANT_1]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           amdgcn.cmpi s_cmp_lt_i32 outs %[[ALLOCA_0]] ins %[[VAL_1]], %[[TO_REG_0]] : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
// CHECK:           amdgcn.cbranch s_cbranch_scc1 %[[ALLOCA_0]] ^bb1 fallthrough(^bb2) : !amdgcn.scc
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_loops_non_const_bounds(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %n_u = aster_utils.assume_uniform %n : i32
  scf.for %i = %c0 to %n_u step %c1 : i32 {
    %iv = lsir.to_reg %i : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}

// -----

func.func @test_non_const_bounds(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-error@+1 {{only thread-uniform loops are supported in this conversion}}
  scf.for %i = %c0 to %n step %c1 : i32 {
    %iv = lsir.to_reg %i : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}

// -----

func.func @test_non_i32_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // expected-error@+1 {{only i32 induction variables are supported in this conversion}}
  scf.for %i = %c0 to %c10 step %c1 {
    %iv = lsir.to_reg %i : index -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}
