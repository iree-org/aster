// RUN: aster-opt %s --aster-inline-execute-region | FileCheck %s

// CHECK-LABEL:   func.func private @callee(index) -> index
func.func private @callee(%x: index) -> index

// CHECK-LABEL:   func.func @test_inline_single_region(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[VAL_0:.*]] = call @callee(%[[ARG0]]) : (index) -> index
// CHECK:           return %[[VAL_0]] : index
// CHECK:         }
func.func @test_inline_single_region(%arg: index) -> index {
  %result = aster_utils.execute_region : index {
    %call = func.call @callee(%arg) : (index) -> index
    aster_utils.yield %call : index
  }
  return %result : index
}

// CHECK-LABEL:   func.func @test_inline_multiple_regions(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[VAL_0:.*]] = call @callee(%[[ARG0]]) : (index) -> index
// CHECK:           %[[VAL_1:.*]] = call @callee(%[[VAL_0]]) : (index) -> index
// CHECK:           return %[[VAL_1]] : index
// CHECK:         }
func.func @test_inline_multiple_regions(%arg: index) -> index {
  %r0 = aster_utils.execute_region : index {
    %c0 = func.call @callee(%arg) : (index) -> index
    aster_utils.yield %c0 : index
  }
  %r1 = aster_utils.execute_region : index {
    %c1 = func.call @callee(%r0) : (index) -> index
    aster_utils.yield %c1 : index
  }
  return %r1 : index
}

// CHECK-LABEL:   func.func @test_inline_region_in_loop(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?xindex>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[CALL_0:.*]] = func.call @callee(%[[VAL_0]]) : (index) -> index
// CHECK:             memref.store %[[CALL_0]], %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_inline_region_in_loop(%lb: index, %ub: index, %mem: memref<?xindex>) {
  %c1 = arith.constant 1 : index
  scf.for %i = %lb to %ub step %c1 {
    %r = aster_utils.execute_region : index {
      %call = func.call @callee(%i) : (index) -> index
      aster_utils.yield %call : index
    }
    memref.store %r, %mem[%i] : memref<?xindex>
  }
  return
}

// CHECK:         func.func private @void_callee()
func.func private @void_callee()

// CHECK-LABEL:   func.func @test_inline_void_region() {
// CHECK:           call @void_callee() : () -> ()
// CHECK:           return
// CHECK:         }
func.func @test_inline_void_region() {
  aster_utils.execute_region {
    func.call @void_callee() : () -> ()
    aster_utils.yield
  }
  return
}

// CHECK-LABEL:   func.func @test_inline_region_with_multiple_ops(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[CONSTANT_0]] : index
// CHECK:           %[[VAL_0:.*]] = call @callee(%[[ADDI_0]]) : (index) -> index
// CHECK:           return %[[VAL_0]] : index
// CHECK:         }
func.func @test_inline_region_with_multiple_ops(%arg: index) -> index {
  %result = aster_utils.execute_region : index {
    %c1 = arith.constant 1 : index
    %add = arith.addi %arg, %c1 : index
    %call = func.call @callee(%add) : (index) -> index
    aster_utils.yield %call : index
  }
  return %result : index
}

// CHECK-LABEL:   func.func @test_inline_region_multiple_results(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> (index, index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[CONSTANT_0]] : index
// CHECK:           return %[[ARG0]], %[[ADDI_0]] : index, index
// CHECK:         }
func.func @test_inline_region_multiple_results(%arg: index) -> (index, index) {
  %r0, %r1 = aster_utils.execute_region : index, index {
    %c1 = arith.constant 1 : index
    %add = arith.addi %arg, %c1 : index
    aster_utils.yield %arg, %add : index, index
  }
  return %r0, %r1 : index, index
}
