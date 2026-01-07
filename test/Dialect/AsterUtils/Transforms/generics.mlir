// RUN: aster-opt %s --inline  --mem2reg --canonicalize | FileCheck %s

func.func private @constexpr_generic_select(%cond: i1, %v0: i32, %v1: f32) -> !aster_utils.any {
  %res = scf.if %cond -> !aster_utils.any {
    %to_any_0 = aster_utils.to_any %v0 : i32
    scf.yield %to_any_0 : !aster_utils.any
  } else {
    %to_any_1 = aster_utils.to_any %v1 : f32
    scf.yield %to_any_1 : !aster_utils.any
  }
  return %res : !aster_utils.any
}

// CHECK-LABEL:   func.func @true_case(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
// CHECK:         }
func.func @true_case(%v0: i32, %v1: f32) -> i32 {
  %true = arith.constant 1 : i1
  %0 = call @constexpr_generic_select(%true, %v0, %v1) : (i1, i32, f32) -> !aster_utils.any
  %res = aster_utils.from_any %0 : i32
  return %res : i32
}

// CHECK-LABEL:   func.func @false_case(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> f32 {
// CHECK:           return %[[ARG1]] : f32
// CHECK:         }
func.func @false_case(%v0: i32, %v1: f32) -> f32 {
  %false = arith.constant 0 : i1
  %0 = call @constexpr_generic_select(%false, %v0, %v1) : (i1, i32, f32) -> !aster_utils.any
  %res = aster_utils.from_any %0 : f32
  return %res : f32
}

func.func private @generic_byref_ret(%cond: i1, %alloc: memref<!aster_utils.any>) {
  %res = scf.if %cond -> !aster_utils.any {
    %v0 = arith.constant 42 : i32
    %to_any_0 = aster_utils.to_any %v0 : i32
    scf.yield %to_any_0 : !aster_utils.any
  } else {
    %v1 = arith.constant -42 : index
    %to_any_1 = aster_utils.to_any %v1 : index
    scf.yield %to_any_1 : !aster_utils.any
  }
  memref.store %res, %alloc[] : memref<!aster_utils.any>
  return
}

// CHECK-LABEL:   func.func @true_case_byref() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           return %[[CONSTANT_0]] : i32
// CHECK:         }
func.func @true_case_byref() -> i32 {
  %alloca = memref.alloca() : memref<!aster_utils.any>
  %true = arith.constant 1 : i1
  call @generic_byref_ret(%true, %alloca) : (i1, memref<!aster_utils.any>) -> ()
  %res = memref.load %alloca[] : memref<!aster_utils.any>
  %res_cast = aster_utils.from_any %res : i32
  return %res_cast : i32
}

// CHECK-LABEL:   func.func @false_case_byref() -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -42 : index
// CHECK:           return %[[CONSTANT_0]] : index
// CHECK:         }
func.func @false_case_byref() -> index {
  %alloca = memref.alloca() : memref<!aster_utils.any>
  %true = arith.constant 0 : i1
  call @generic_byref_ret(%true, %alloca) : (i1, memref<!aster_utils.any>) -> ()
  %res = memref.load %alloca[] : memref<!aster_utils.any>
  %res_cast = aster_utils.from_any %res : index
  return %res_cast : index
}
