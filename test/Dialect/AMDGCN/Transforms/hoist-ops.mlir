// RUN: aster-opt %s --pass-pipeline="builtin.module(func.func(aster-hoist-ops))" | FileCheck %s

// CHECK-LABEL:   func.func @test_func(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: i1) -> (!amdgcn.sgpr, !amdgcn.sgpr) {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[LOAD_ARG_0:.*]] = amdgcn.load_arg 0 : !amdgcn.sgpr
// CHECK-DAG:       %[[LOAD_ARG_1:.*]] = amdgcn.load_arg 2 : !amdgcn.sgpr
// CHECK-DAG:       %[[LOAD_ARG_2:.*]] = amdgcn.load_arg 1 : !amdgcn.sgpr
// CHECK-DAG:       %[[THREAD_ID_0:.*]] = amdgcn.thread_id  x : !amdgcn.vgpr
// CHECK-DAG:       %[[BLOCK_ID_0:.*]] = amdgcn.block_id  y : !amdgcn.sgpr
// CHECK-DAG:       %[[THREAD_ID_1:.*]] = amdgcn.thread_id  y : !amdgcn.vgpr
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK:             scf.if %[[ARG1]] {
// CHECK:               amdgcn.test_inst ins %[[LOAD_ARG_0]], %[[LOAD_ARG_2]], %[[THREAD_ID_0]] : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> ()
// CHECK:             } else {
// CHECK:               amdgcn.test_inst ins %[[LOAD_ARG_0]], %[[LOAD_ARG_1]] : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           cf.cond_br %[[ARG1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[THREAD_ID_1]] : (!amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[BLOCK_ID_0]], %[[THREAD_ID_0]] : (!amdgcn.sgpr, !amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         }
func.func @test_func(%arg0: index, %arg1: i1) -> (!amdgcn.sgpr, !amdgcn.sgpr) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %arg0 step %c1 {
    %5 = amdgcn.load_arg 0 : !amdgcn.sgpr
    scf.if %arg1 {
      %6 = amdgcn.load_arg 1 : !amdgcn.sgpr
      %7 = amdgcn.thread_id  x : !amdgcn.vgpr
      amdgcn.test_inst ins %5, %6, %7 : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> ()
    } else {
      %6 = amdgcn.load_arg 2 : !amdgcn.sgpr
      amdgcn.test_inst ins %5, %6 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    }
  }
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %1 = amdgcn.thread_id  y : !amdgcn.vgpr
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %2 = amdgcn.block_id  y : !amdgcn.sgpr
  %3 = amdgcn.thread_id  x : !amdgcn.vgpr
  amdgcn.test_inst ins %2, %3 : (!amdgcn.sgpr, !amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %4 = amdgcn.alloca : !amdgcn.sgpr
  return %0, %4 : !amdgcn.sgpr, !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_func_allocated(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: i1) -> (!amdgcn.sgpr<1>, !amdgcn.sgpr<1>) {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[LOAD_ARG_0:.*]] = amdgcn.load_arg 0 : !amdgcn.sgpr
// CHECK-DAG:       %[[LOAD_ARG_1:.*]] = amdgcn.load_arg 2 : !amdgcn.sgpr
// CHECK-DAG:       %[[LOAD_ARG_2:.*]] = amdgcn.load_arg 1 : !amdgcn.sgpr
// CHECK-DAG:       %[[THREAD_ID_0:.*]] = amdgcn.thread_id  x : !amdgcn.vgpr
// CHECK-DAG:       %[[BLOCK_ID_0:.*]] = amdgcn.block_id  y : !amdgcn.sgpr
// CHECK-DAG:       %[[THREAD_ID_1:.*]] = amdgcn.thread_id  y : !amdgcn.vgpr
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK:             scf.if %[[ARG1]] {
// CHECK:               amdgcn.test_inst ins %[[LOAD_ARG_0]], %[[LOAD_ARG_2]], %[[THREAD_ID_0]] : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> ()
// CHECK:             } else {
// CHECK:               amdgcn.test_inst ins %[[LOAD_ARG_0]], %[[LOAD_ARG_1]] : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           cf.cond_br %[[ARG1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[THREAD_ID_1]] : (!amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[BLOCK_ID_0]], %[[THREAD_ID_0]] : (!amdgcn.sgpr, !amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_0]] : !amdgcn.sgpr<1>, !amdgcn.sgpr<1>
// CHECK:         }
func.func @test_func_allocated(%arg0: index, %arg1: i1) -> (!amdgcn.sgpr<1>,!amdgcn.sgpr<1>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr<1>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %arg0 step %c1 {
    %5 = amdgcn.load_arg 0 : !amdgcn.sgpr
    scf.if %arg1 {
      %6 = amdgcn.load_arg 1 : !amdgcn.sgpr
      %7 = amdgcn.thread_id  x : !amdgcn.vgpr
      amdgcn.test_inst ins %5, %6, %7 : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> ()
    } else {
      %6 = amdgcn.load_arg 2 : !amdgcn.sgpr
      amdgcn.test_inst ins %5, %6 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    }
  }
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %1 = amdgcn.thread_id  y : !amdgcn.vgpr
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %2 = amdgcn.block_id  y : !amdgcn.sgpr
  %3 = amdgcn.thread_id  x : !amdgcn.vgpr
  amdgcn.test_inst ins %2, %3 : (!amdgcn.sgpr, !amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %4 = amdgcn.alloca : !amdgcn.sgpr<1>
  return %0, %4 : !amdgcn.sgpr<1>, !amdgcn.sgpr<1>
}
