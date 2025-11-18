// RUN: aster-opt %s --aster-wrap-calls-with-execute-region | FileCheck %s

// CHECK-LABEL:   func.func private @callee(index) -> index
func.func private @callee(%x: index) -> index

// CHECK-LABEL:   func.func @test_wrap_single_call(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = aster_utils.execute_region : index {
// CHECK:             %[[CALL_0:.*]] = func.call @callee(%[[ARG0]]) : (index) -> index
// CHECK:             aster_utils.yield %[[CALL_0]] : index
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : index
// CHECK:         }
func.func @test_wrap_single_call(%arg: index) -> index {
  %result = func.call @callee(%arg) : (index) -> index
  return %result : index
}

// CHECK-LABEL:   func.func @test_wrap_multiple_calls(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = aster_utils.execute_region : index {
// CHECK:             %[[CALL_0:.*]] = func.call @callee(%[[ARG0]]) : (index) -> index
// CHECK:             aster_utils.yield %[[CALL_0]] : index
// CHECK:           }
// CHECK:           %[[EXECUTE_REGION_1:.*]] = aster_utils.execute_region : index {
// CHECK:             %[[CALL_1:.*]] = func.call @callee(%[[EXECUTE_REGION_0]]) : (index) -> index
// CHECK:             aster_utils.yield %[[CALL_1]] : index
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_1]] : index
// CHECK:         }
func.func @test_wrap_multiple_calls(%arg: index) -> index {
  %r0 = func.call @callee(%arg) : (index) -> index
  %r1 = func.call @callee(%r0) : (index) -> index
  return %r1 : index
}

// CHECK-LABEL:   func.func @test_wrap_call_in_loop(
// CHECK-SAME:      %[[ARG0:.*]]: index,  %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?xindex>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXECUTE_REGION_0:.*]] = aster_utils.execute_region : index {
// CHECK:               %[[CALL_0:.*]] = func.call @callee(%[[VAL_0]]) : (index) -> index
// CHECK:               aster_utils.yield %[[CALL_0]] : index
// CHECK:             }
// CHECK:             memref.store %[[EXECUTE_REGION_0]], %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_wrap_call_in_loop(%lb: index, %ub: index, %mem: memref<?xindex>) {
  %c1 = arith.constant 1 : index
  scf.for %i = %lb to %ub step %c1 {
    %r = func.call @callee(%i) : (index) -> index
    memref.store %r, %mem[%i] : memref<?xindex>
  }
  return
}

// CHECK-LABEL:   func.func @test_no_rewrap_already_wrapped(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = aster_utils.execute_region : index {
// CHECK:             %[[CALL_0:.*]] = func.call @callee(%[[ARG0]]) : (index) -> index
// CHECK:             aster_utils.yield %[[CALL_0]] : index
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : index
// CHECK:         }
func.func @test_no_rewrap_already_wrapped(%arg: index) -> index {
  %result = aster_utils.execute_region : index {
    %call = func.call @callee(%arg) : (index) -> index
    aster_utils.yield %call : index
  }
  return %result : index
}

// CHECK:         func.func private @void_callee()
func.func private @void_callee()

// CHECK-LABEL:   func.func @test_wrap_void_call() {
// CHECK:           aster_utils.execute_region {
// CHECK:             func.call @void_callee() : () -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_wrap_void_call() {
  func.call @void_callee() : () -> ()
  return
}
