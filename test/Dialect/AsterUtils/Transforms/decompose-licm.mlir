// RUN: aster-opt %s --aster-decompose-by-loop-invariant | FileCheck %s


func.func private @test(index)
// CHECK-LABEL:   func.func @loop_invariant_subexpr(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG2]], %[[CONSTANT_1]] : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]] {
// CHECK:             %[[ADDI_0:.*]] = aster_utils.addi %[[VAL_0]], %[[MULI_0]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             func.call @test(%[[ADDI_1]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @loop_invariant_subexpr(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = aster_utils.muli %arg2, %c8 : index
  scf.for %arg3 = %arg0 to %arg1 step %arg2 {
    %1 = aster_utils.addi %arg3, %0 : index
    %2 = arith.addi %1, %c0 : index
    func.call @test(%2) : (index) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @loop_invariant_constant(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 42 : index
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG2]], %[[CONSTANT_1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]] {
// CHECK:             %[[ADDI_1:.*]] = aster_utils.addi %[[VAL_0]], %[[PASSTHROUGH_0]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CONSTANT_0]] : index
// CHECK:             func.call @test(%[[ADDI_2]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @loop_invariant_constant(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %0 = aster_utils.addi %arg2, %c42 : index
  %1 = aster_utils.passthrough %0 tag = "__decompose_ops__" : index
  scf.for %arg3 = %arg0 to %arg1 step %arg2 {
    %2 = aster_utils.addi %arg3, %1 : index
    %3 = arith.addi %2, %c0 : index
    func.call @test(%3) : (index) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @none_hoistable(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]] {
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[ADDI_1:.*]] = aster_utils.addi %[[VAL_0]], %[[ADDI_0]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CONSTANT_0]] : index
// CHECK:             func.call @test(%[[ADDI_2]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @none_hoistable(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %arg0 to %arg1 step %arg2 {
    %0 = arith.addi %arg3, %c0 : index
    %1 = aster_utils.addi %arg3, %0 : index
    %2 = arith.addi %1, %c0 : index
    func.call @test(%2) : (index) -> ()
  }
  return
}
