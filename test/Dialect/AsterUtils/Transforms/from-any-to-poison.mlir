// RUN: aster-opt %s --aster-from-any-to-poison | FileCheck %s

// CHECK-LABEL:   func.func @test_from_any_to_poison(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any) -> i32 {
// CHECK:           %[[POISON:.*]] = ub.poison : i32
// CHECK:           return %[[POISON]] : i32
// CHECK:         }
func.func @test_from_any_to_poison(%arg: !aster_utils.any) -> i32 {
  %result = aster_utils.from_any %arg : i32
  return %result : i32
}

// CHECK-LABEL:   func.func @test_from_any_to_poison_f32(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any) -> f32 {
// CHECK:           %[[POISON:.*]] = ub.poison : f32
// CHECK:           return %[[POISON]] : f32
// CHECK:         }
func.func @test_from_any_to_poison_f32(%arg: !aster_utils.any) -> f32 {
  %result = aster_utils.from_any %arg : f32
  return %result : f32
}

// CHECK-LABEL:   func.func @test_from_any_to_poison_vector(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any) -> vector<4xf32> {
// CHECK:           %[[POISON:.*]] = ub.poison : vector<4xf32>
// CHECK:           return %[[POISON]] : vector<4xf32>
// CHECK:         }
func.func @test_from_any_to_poison_vector(%arg: !aster_utils.any) -> vector<4xf32> {
  %result = aster_utils.from_any %arg : vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL:   func.func @test_multiple_from_any(
// CHECK-SAME:      %[[ARG0:.*]]: !aster_utils.any,
// CHECK-SAME:      %[[ARG1:.*]]: !aster_utils.any) -> (i32, f64) {
// CHECK:           %[[POISON0:.*]] = ub.poison : i32
// CHECK:           %[[POISON1:.*]] = ub.poison : f64
// CHECK:           return %[[POISON0]], %[[POISON1]] : i32, f64
// CHECK:         }
func.func @test_multiple_from_any(%arg0: !aster_utils.any, %arg1: !aster_utils.any) -> (i32, f64) {
  %r0 = aster_utils.from_any %arg0 : i32
  %r1 = aster_utils.from_any %arg1 : f64
  return %r0, %r1 : i32, f64
}


// CHECK-LABEL:   func.func @test_with_origin(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           return %[[ARG0]] : i32
// CHECK:         }
func.func @test_with_origin(%arg0: i32) -> i32 {
  %0 = aster_utils.to_any %arg0 : i32
  %1 = aster_utils.from_any %0 : i32
  return %1 : i32
}

// CHECK-LABEL:   func.func @test_with_mismatched_origin(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> f32 {
// CHECK:           %[[POISON:.*]] = ub.poison : f32
// CHECK:           return %[[POISON]] : f32
// CHECK:         }
func.func @test_with_mismatched_origin(%arg0: i32) -> f32 {
  %0 = aster_utils.to_any %arg0 : i32
  %1 = aster_utils.from_any %0 : f32
  return %1 : f32
}
