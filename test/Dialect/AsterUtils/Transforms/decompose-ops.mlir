// RUN: aster-opt %s --aster-decompose-ops | FileCheck %s


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

// CHECK-LABEL:   func.func @test_three_way_addi(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_3:.*]] = aster_utils.addi %[[ARG4]], %[[PASSTHROUGH_0]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[ADDI_3]] : index, index, index
// CHECK:         }
func.func @test_three_way_addi(%a: index, %b: index, %c: index,
                               %d: index, %e: index) -> (index, index, index) {
  %x = aster_utils.addi %a, %b, %c : index
  %y = aster_utils.addi %a, %b, %d : index
  %z = aster_utils.addi %a, %b, %e : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_no_cse_single_op(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_no_cse_single_op(%a: index, %b: index) -> index {
  %r = aster_utils.addi %a, %b : index
  return %r : index
}

// CHECK-LABEL:   func.func @test_no_cse_disjoint(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[ARG3]] : index
// CHECK:           return %[[ADDI_0]], %[[ADDI_1]] : index, index
// CHECK:         }
func.func @test_no_cse_disjoint(%a: index, %b: index,
                                %c: index, %d: index) -> (index, index) {
  %x = aster_utils.addi %a, %b : index
  %y = aster_utils.addi %c, %d : index
  return %x, %y : index, index
}

// CHECK-LABEL:   func.func @test_identical_ops(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> (index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           return %[[PASSTHROUGH_0]], %[[PASSTHROUGH_0]] : index, index
// CHECK:         }
func.func @test_identical_ops(%a: index, %b: index) -> (index, index) {
  %x = aster_utils.addi %a, %b : index
  %y = aster_utils.addi %a, %b : index
  return %x, %y : index, index
}

// CHECK-LABEL:   func.func @test_two_level(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG2]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[ADDI_1]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG1]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_2]], %[[PASSTHROUGH_1]], %[[PASSTHROUGH_0]] : index, index, index
// CHECK:         }
func.func @test_two_level(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %x = aster_utils.addi %a, %b, %c, %d : index
  %y = aster_utils.addi %a, %c, %d : index
  %z = aster_utils.addi %a, %c : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_three_way_muli(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> (index, index, index) {
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[MULI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_2:.*]] = aster_utils.muli %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_3:.*]] = aster_utils.muli %[[ARG4]], %[[PASSTHROUGH_0]] : index
// CHECK:           return %[[MULI_1]], %[[MULI_2]], %[[MULI_3]] : index, index, index
// CHECK:         }
func.func @test_three_way_muli(%a: index, %b: index, %c: index,
                               %d: index, %e: index) -> (index, index, index) {
  %x = aster_utils.muli %a, %b, %c : index
  %y = aster_utils.muli %a, %b, %d : index
  %z = aster_utils.muli %a, %b, %e : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_addi_muli_independent(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index) -> (index, index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[MULI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[ARG4]], %[[PASSTHROUGH_1]] : index
// CHECK:           %[[MULI_2:.*]] = aster_utils.muli %[[ARG5]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[MULI_1]], %[[MULI_2]] : index, index, index, index
// CHECK:         }
func.func @test_addi_muli_independent(%a: index, %b: index, %c: index,
                                      %d: index, %e: index, %f: index) -> (index, index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d : index
  %xm = aster_utils.muli %a, %b, %e : index
  %ym = aster_utils.muli %a, %b, %f : index
  return %xa, %ya, %xm, %ym : index, index, index, index
}

// CHECK-LABEL:   func.func @test_muli_not_cse_single(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[MULI_0]] : index, index, index
// CHECK:         }
func.func @test_muli_not_cse_single(%a: index, %b: index, %c: index,
                                    %d: index) -> (index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d : index
  %xm = aster_utils.muli %a, %b : index
  return %xa, %ya, %xm : index, index, index
}

// CHECK-LABEL:   func.func @test_chained_cse(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ADDI_1]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[ADDI_2]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_3:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_1]] : index
// CHECK:           %[[ADDI_4:.*]] = aster_utils.addi %[[ADDI_3]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_3]], %[[ADDI_4]] : index, index, index
// CHECK:         }
func.func @test_chained_cse(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d, %xa : index
  %xm = aster_utils.addi %a, %b, %xa, %ya : index
  return %xa, %ya, %xm : index, index, index
}
