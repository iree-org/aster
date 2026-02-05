// RUN: aster-opt %s --amdgcn-convert-scf-control-flow --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @test_uniform_loops_const_bounds() {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^bb1(%[[C0]] : i32), ^bb2
// CHECK:         ^bb1(%[[IV:.*]]: i32):
// CHECK:           %[[TO_REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^bb1(%[[IV_NEXT]] : i32), ^bb2
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[N_U:.*]] = aster_utils.assume_uniform %[[ARG0]] : i32
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[N_U]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^bb1(%[[C0]] : i32), ^bb2
// CHECK:         ^bb1(%[[IV:.*]]: i32):
// CHECK:           %[[TO_REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[N_U]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^bb1(%[[IV_NEXT]] : i32), ^bb2
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

func.func @test_index_loop_unsupported() {
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
