// RUN: aster-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @test_fold_from_to_any_i32(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
// CHECK:         }
func.func @test_fold_from_to_any_i32(%arg: i32) -> i32 {
  %any = aster_utils.to_any %arg : i32
  %result = aster_utils.from_any %any : i32
  return %result : i32
}

// CHECK-LABEL:   func.func @test_fold_from_to_any_f32(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> f32 {
// CHECK:           return %[[ARG0]] : f32
// CHECK:         }
func.func @test_fold_from_to_any_f32(%arg: f32) -> f32 {
  %any = aster_utils.to_any %arg : f32
  %result = aster_utils.from_any %any : f32
  return %result : f32
}

// CHECK-LABEL:   func.func @test_no_fold_type_mismatch(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i64 {
// CHECK:           %[[TO_ANY_0:.*]] = aster_utils.to_any %[[ARG0]] : i32
// CHECK:           %[[FROM_ANY_0:.*]] = aster_utils.from_any %[[TO_ANY_0]] : i64
// CHECK:           return %[[FROM_ANY_0]] : i64
// CHECK:         }
func.func @test_no_fold_type_mismatch(%arg: i32) -> i64 {
  %any = aster_utils.to_any %arg : i32
  %result = aster_utils.from_any %any : i64
  return %result : i64
}

// CHECK-LABEL:   func.func @test_no_fold_to_from_any_info_loss(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any) -> !aster_utils.any {
// CHECK:           %[[FROM_ANY_0:.*]] = aster_utils.from_any %[[ARG0]] : i32
// CHECK:           %[[TO_ANY_0:.*]] = aster_utils.to_any %[[FROM_ANY_0]] : i32
// CHECK:           return %[[TO_ANY_0]] : !aster_utils.any
// CHECK:         }
func.func @test_no_fold_to_from_any_info_loss(%arg: !aster_utils.any) -> !aster_utils.any {
  %val = aster_utils.from_any %arg : i32
  %result = aster_utils.to_any %val : i32
  return %result : !aster_utils.any
}

// CHECK-LABEL:   func.func @test_no_fold_to_from_type_mismatch(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any) -> !aster_utils.any {
// CHECK:           %[[FROM_ANY_0:.*]] = aster_utils.from_any %[[ARG0]] : i32
// CHECK:           %[[TO_ANY_0:.*]] = aster_utils.to_any %[[FROM_ANY_0]] : i32
// CHECK:           return %[[TO_ANY_0]] : !aster_utils.any
// CHECK:         }
func.func @test_no_fold_to_from_type_mismatch(%arg: !aster_utils.any) -> !aster_utils.any {
  %val = aster_utils.from_any %arg : i32
  %result = aster_utils.to_any %val : i32
  return %result : !aster_utils.any
}

// CHECK-LABEL:   func.func @test_fold_chain(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
// CHECK:         }
func.func @test_fold_chain(%arg: i32) -> i32 {
  %any1 = aster_utils.to_any %arg : i32
  %val1 = aster_utils.from_any %any1 : i32
  %any2 = aster_utils.to_any %val1 : i32
  %val2 = aster_utils.from_any %any2 : i32
  return %val2 : i32
}
