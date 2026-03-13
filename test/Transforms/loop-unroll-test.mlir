// RUN: aster-opt --aster-loop-unroll="unroll-factors=2,3,4,5,6" %s | FileCheck %s


func.func private @test(%i : index)

// CHECK-LABEL:   func.func @trip_div_by_2() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 14 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
// CHECK:             func.call @test(%[[VAL_0]]) : (index) -> ()
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_4]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[MULI_0]] : index
// CHECK:             func.call @test(%[[ADDI_0]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @trip_div_by_2() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c14 = arith.constant 14: index
  scf.for %i = %c0 to %c14 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @trip_factor_equals() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           call @test(%[[CONSTANT_0]]) : (index) -> ()
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_4]] : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[MULI_0]] : index
// CHECK:           call @test(%[[ADDI_0]]) : (index) -> ()
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_5]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_0]], %[[MULI_1]] : index
// CHECK:           call @test(%[[ADDI_1]]) : (index) -> ()
// CHECK:           return
// CHECK:         }
func.func @trip_factor_equals() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @trip_divides_largest() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 40 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 5 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
// CHECK:             func.call @test(%[[VAL_0]]) : (index) -> ()
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_4]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[MULI_0]] : index
// CHECK:             func.call @test(%[[ADDI_0]]) : (index) -> ()
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_5]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[MULI_1]] : index
// CHECK:             func.call @test(%[[ADDI_1]]) : (index) -> ()
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_6]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_0]], %[[MULI_2]] : index
// CHECK:             func.call @test(%[[ADDI_2]]) : (index) -> ()
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant 4 : index
// CHECK:             %[[MULI_3:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_7]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_0]], %[[MULI_3]] : index
// CHECK:             func.call @test(%[[ADDI_3]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @trip_divides_largest() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  scf.for %i = %c0 to %c40 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @trip_7_not_divisible() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 7 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:             func.call @test(%[[VAL_0]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @trip_7_not_divisible() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  scf.for %i = %c0 to %c7 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @trip_12_div_by_2_3_4_6() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 12 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 6 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
// CHECK:             func.call @test(%[[VAL_0]]) : (index) -> ()
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_4]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[MULI_0]] : index
// CHECK:             func.call @test(%[[ADDI_0]]) : (index) -> ()
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_5]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[MULI_1]] : index
// CHECK:             func.call @test(%[[ADDI_1]]) : (index) -> ()
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_6]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_0]], %[[MULI_2]] : index
// CHECK:             func.call @test(%[[ADDI_2]]) : (index) -> ()
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant 4 : index
// CHECK:             %[[MULI_3:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_7]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_0]], %[[MULI_3]] : index
// CHECK:             func.call @test(%[[ADDI_3]]) : (index) -> ()
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant 5 : index
// CHECK:             %[[MULI_4:.*]] = arith.muli %[[CONSTANT_1]], %[[CONSTANT_8]] : index
// CHECK:             %[[ADDI_4:.*]] = arith.addi %[[VAL_0]], %[[MULI_4]] : index
// CHECK:             func.call @test(%[[ADDI_4]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @trip_12_div_by_2_3_4_6() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  scf.for %i = %c0 to %c12 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @dynamic_trip_count(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK:             func.call @test(%[[VAL_0]]) : (index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @dynamic_trip_count(%n : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL:   func.func @unroll_factor_attr_single() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 14 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 7 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
func.func @unroll_factor_attr_single() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c14 = arith.constant 14 : index
  scf.for %i = %c0 to %c14 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  } {unroll_factor = 7 : i64}
  return
}

// CHECK-LABEL:   func.func @unroll_factor_attr_array() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 14 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 7 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
func.func @unroll_factor_attr_array() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c14 = arith.constant 14 : index
  scf.for %i = %c0 to %c14 step %c1 {
    func.call @test(%i) : (index) -> ()
    scf.yield
  } {unroll_factor = array<i64: 2, 7>}
  return
}
