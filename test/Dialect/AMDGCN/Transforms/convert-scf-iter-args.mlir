// RUN: aster-opt %s --amdgcn-convert-scf-control-flow | FileCheck %s

// Test scf.for with a single iter_arg (accumulator pattern)
// Uses index_cast to convert i32 bounds to index (required by scf.for with iter_args)
// CHECK-LABEL:   func.func @test_iter_args_single(
// CHECK-SAME:      %[[INIT:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[LB:.*]] = arith.index_cast %[[C0]] : i32 to index
// CHECK:           %[[UB:.*]] = arith.index_cast %[[C10]] : i32 to index
// CHECK:           %[[STEP:.*]] = arith.index_cast %[[C1]] : i32 to index
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[LB]], %[[UB]] : index
// CHECK:           cf.cond_br %[[INIT_CMP]], ^bb1(%[[LB]], %[[INIT]] : index, i32), ^bb2(%[[INIT]] : i32)
// CHECK:         ^bb1(%[[IV:.*]]: index, %[[ACC:.*]]: i32):
// CHECK:           %[[NEW_ACC:.*]] = arith.addi %[[ACC]], %[[C1]] : i32
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[STEP]] : index
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[UB]] : index
// CHECK:           cf.cond_br %[[BACK_CMP]], ^bb1(%[[IV_NEXT]], %[[NEW_ACC]] : index, i32), ^bb2(%[[NEW_ACC]] : i32)
// CHECK:         ^bb2(%[[RESULT:.*]]: i32):
// CHECK:           return %[[RESULT]] : i32
// CHECK:         }
func.func @test_iter_args_single(%init: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %lb = arith.index_cast %c0 : i32 to index
  %ub = arith.index_cast %c10 : i32 to index
  %step = arith.index_cast %c1 : i32 to index
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
    %new_acc = arith.addi %acc, %c1 : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

// Test scf.for with multiple iter_args
// CHECK-LABEL:   func.func @test_iter_args_multiple(
// CHECK-SAME:      %[[INIT_A:.*]]: i32, %[[INIT_B:.*]]: i32) -> (i32, i32) {
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[INIT_A]], %[[INIT_B]] : index, i32, i32), ^bb2(%[[INIT_A]], %[[INIT_B]] : i32, i32)
// CHECK:         ^bb1(%[[IV:.*]]: index, %[[ACC_A:.*]]: i32, %[[ACC_B:.*]]: i32):
// CHECK:           %[[NEW_A:.*]] = arith.addi %[[ACC_A]], %{{.*}} : i32
// CHECK:           %[[NEW_B:.*]] = arith.muli %[[ACC_B]], %{{.*}} : i32
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[NEW_A]], %[[NEW_B]] : index, i32, i32), ^bb2(%[[NEW_A]], %[[NEW_B]] : i32, i32)
// CHECK:         ^bb2(%[[RES_A:.*]]: i32, %[[RES_B:.*]]: i32):
// CHECK:           return %[[RES_A]], %[[RES_B]] : i32, i32
// CHECK:         }
func.func @test_iter_args_multiple(%init_a: i32, %init_b: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c5 = arith.constant 5 : i32
  %lb = arith.index_cast %c0 : i32 to index
  %ub = arith.index_cast %c5 : i32 to index
  %step = arith.index_cast %c1 : i32 to index
  %result_a, %result_b = scf.for %i = %lb to %ub step %step iter_args(%acc_a = %init_a, %acc_b = %init_b) -> (i32, i32) {
    %new_a = arith.addi %acc_a, %c1 : i32
    %new_b = arith.muli %acc_b, %c2 : i32
    scf.yield %new_a, %new_b : i32, i32
  }
  return %result_a, %result_b : i32, i32
}

// Test scf.for with VGPR register type iter_arg
// CHECK-LABEL:   func.func @test_iter_args_vgpr(
// CHECK-SAME:      %[[INIT:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[OUT:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[INIT]] : index, !amdgcn.vgpr), ^bb2(%[[INIT]] : !amdgcn.vgpr)
// CHECK:         ^bb1(%[[IV:.*]]: index, %[[ACC:.*]]: !amdgcn.vgpr):
// CHECK:           %[[NEW_ACC:.*]] = amdgcn.test_inst outs %[[OUT]] ins %[[ACC]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[NEW_ACC]] : index, !amdgcn.vgpr), ^bb2(%[[NEW_ACC]] : !amdgcn.vgpr)
// CHECK:         ^bb2(%[[RESULT:.*]]: !amdgcn.vgpr):
// CHECK:           return %[[RESULT]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_iter_args_vgpr(%init: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %lb = arith.index_cast %c0 : i32 to index
  %ub = arith.index_cast %c8 : i32 to index
  %step = arith.index_cast %c1 : i32 to index
  %out = amdgcn.alloca : !amdgcn.vgpr
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (!amdgcn.vgpr) {
    %new_acc = amdgcn.test_inst outs %out ins %acc : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %new_acc : !amdgcn.vgpr
  }
  return %result : !amdgcn.vgpr
}

// Test zero iterations case - init values should be returned directly
// CHECK-LABEL:   func.func @test_iter_args_zero_iterations(
// CHECK-SAME:      %[[INIT:.*]]: i32) -> i32 {
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[INIT]] : index, i32), ^bb2(%[[INIT]] : i32)
// CHECK:         ^bb1
// CHECK:         ^bb2(%[[RESULT:.*]]: i32):
// CHECK:           return %[[RESULT]] : i32
func.func @test_iter_args_zero_iterations(%init: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %lb = arith.index_cast %c5 : i32 to index
  %ub = arith.index_cast %c5 : i32 to index
  %step = arith.index_cast %c1 : i32 to index
  // Loop from 5 to 5 - zero iterations
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
    %new_acc = arith.addi %acc, %c1 : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

// Test scf.for that yields iter_args directly (swap/rotation pattern).
// This exercises the case where yield operands reference the loop's own
// block arguments, which must be remapped after inlining.
// CHECK-LABEL:   func.func @test_iter_args_swap(
// CHECK-SAME:      %[[INIT_A:.*]]: i32, %[[INIT_B:.*]]: i32) -> (i32, i32) {
// CHECK:           cf.cond_br %{{.*}}, ^bb1(%{{.*}}, %[[INIT_A]], %[[INIT_B]] : index, i32, i32), ^bb2(%[[INIT_A]], %[[INIT_B]] : i32, i32)
// CHECK:         ^bb1(%[[IV:.*]]: index, %[[A:.*]]: i32, %[[B:.*]]: i32):
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %{{.*}} : index
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %{{.*}} : index
// CHECK:           cf.cond_br %[[CMP]], ^bb1(%[[IV_NEXT]], %[[B]], %[[A]] : index, i32, i32), ^bb2(%[[B]], %[[A]] : i32, i32)
// CHECK:         ^bb2(%[[RES_A:.*]]: i32, %[[RES_B:.*]]: i32):
// CHECK:           return %[[RES_A]], %[[RES_B]] : i32, i32
// CHECK:         }
func.func @test_iter_args_swap(%init_a: i32, %init_b: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %lb = arith.index_cast %c0 : i32 to index
  %ub = arith.index_cast %c4 : i32 to index
  %step = arith.index_cast %c1 : i32 to index
  %result_a, %result_b = scf.for %i = %lb to %ub step %step
      iter_args(%a = %init_a, %b = %init_b) -> (i32, i32) {
    // Swap: yield iter_args in reverse order
    scf.yield %b, %a : i32, i32
  }
  return %result_a, %result_b : i32, i32
}
