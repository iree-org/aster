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

//===----------------------------------------------------------------------===//
// StructExtractOp canonicalization tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @test_fold_struct_extract_of_create_single(
// CHECK-SAME:      %[[X:.*]]: i32, %[[Y:.*]]: f32) -> i32 {
// CHECK-NOT:       aster_utils.struct_create
// CHECK-NOT:       aster_utils.struct_extract
// CHECK:           return %[[X]] : i32
// CHECK:         }
func.func @test_fold_struct_extract_of_create_single(%x: i32, %y: f32) -> i32 {
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  %extracted = aster_utils.struct_extract %s ["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
  return %extracted : i32
}

// CHECK-LABEL:   func.func @test_fold_struct_extract_of_create_second_field(
// CHECK-SAME:      %[[X:.*]]: i32, %[[Y:.*]]: f32) -> f32 {
// CHECK-NOT:       aster_utils.struct_create
// CHECK-NOT:       aster_utils.struct_extract
// CHECK:           return %[[Y]] : f32
// CHECK:         }
func.func @test_fold_struct_extract_of_create_second_field(%x: i32, %y: f32) -> f32 {
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  %extracted = aster_utils.struct_extract %s ["y"] : !aster_utils.struct<x: i32, y: f32> -> f32
  return %extracted : f32
}

// CHECK-LABEL:   func.func @test_fold_struct_extract_of_create_multiple(
// CHECK-SAME:      %[[X:.*]]: i32, %[[Y:.*]]: f32) -> (i32, f32) {
// CHECK-NOT:       aster_utils.struct_create
// CHECK-NOT:       aster_utils.struct_extract
// CHECK:           return %[[X]], %[[Y]] : i32, f32
// CHECK:         }
func.func @test_fold_struct_extract_of_create_multiple(%x: i32, %y: f32) -> (i32, f32) {
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  %ex, %ey = aster_utils.struct_extract %s ["x", "y"] : !aster_utils.struct<x: i32, y: f32> -> i32, f32
  return %ex, %ey : i32, f32
}

// CHECK-LABEL:   func.func @test_fold_struct_extract_of_create_reorder(
// CHECK-SAME:      %[[X:.*]]: i32, %[[Y:.*]]: f32) -> (f32, i32) {
// CHECK-NOT:       aster_utils.struct_create
// CHECK-NOT:       aster_utils.struct_extract
// CHECK:           return %[[Y]], %[[X]] : f32, i32
// CHECK:         }
func.func @test_fold_struct_extract_of_create_reorder(%x: i32, %y: f32) -> (f32, i32) {
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  // Extract in different order than the struct definition
  %ey, %ex = aster_utils.struct_extract %s ["y", "x"] : !aster_utils.struct<x: i32, y: f32> -> f32, i32
  return %ey, %ex : f32, i32
}

// Test that we don't fold when the struct is not from a create op
// CHECK-LABEL:   func.func @test_no_fold_struct_extract_arg(
// CHECK-SAME:      %[[S:.*]]: !aster_utils.struct<x: i32, y: f32>) -> i32 {
// CHECK:           %[[EXTRACTED:.*]] = aster_utils.struct_extract %[[S]] ["x"]
// CHECK:           return %[[EXTRACTED]] : i32
// CHECK:         }
func.func @test_no_fold_struct_extract_arg(%s: !aster_utils.struct<x: i32, y: f32>) -> i32 {
  %x = aster_utils.struct_extract %s ["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
  return %x : i32
}

//===----------------------------------------------------------------------===//
// AssumeRangeOp canonicalization tests
//===----------------------------------------------------------------------===//

// No-bounds assume_range is NOT folded to identity (the new fold only handles
// constant-dynamic-to-static conversion).
// CHECK-LABEL:   func.func @test_assume_range_no_bounds_persists(
// CHECK-SAME:      %[[X:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_assume_range_no_bounds_persists(%x: i64) -> i64 {
  %0 = aster_utils.assume_range %x : i64
  return %0 : i64
}

// Fold constant dynamic min to static min.
// CHECK-LABEL:   func.func @test_fold_constant_dynamic_min(
// CHECK-SAME:      %[[X:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min 0
// CHECK-SAME:        max 1024 : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_fold_constant_dynamic_min(%x: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %0 = aster_utils.assume_range %x min %c0 max 1024 : i64
  return %0 : i64
}

// Fold constant dynamic max to static max.
// CHECK-LABEL:   func.func @test_fold_constant_dynamic_max(
// CHECK-SAME:      %[[X:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min 0
// CHECK-SAME:        max 1024 : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_fold_constant_dynamic_max(%x: i64) -> i64 {
  %c1024 = arith.constant 1024 : i64
  %0 = aster_utils.assume_range %x min 0 max %c1024 : i64
  return %0 : i64
}

// Fold both constant dynamic bounds to static.
// CHECK-LABEL:   func.func @test_fold_both_constant_dynamic(
// CHECK-SAME:      %[[X:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min 0
// CHECK-SAME:        max 1024 : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_fold_both_constant_dynamic(%x: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1024 = arith.constant 1024 : i64
  %0 = aster_utils.assume_range %x min %c0 max %c1024 : i64
  return %0 : i64
}

// Non-constant dynamic bounds should NOT be folded.
// CHECK-LABEL:   func.func @test_no_fold_dynamic_bounds(
// CHECK-SAME:      %[[X:.*]]: i64, %[[LO:.*]]: i64, %[[HI:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min %[[LO]]
// CHECK-SAME:        max %[[HI]] : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_no_fold_dynamic_bounds(%x: i64, %lo: i64, %hi: i64) -> i64 {
  %0 = aster_utils.assume_range %x min %lo max %hi : i64
  return %0 : i64
}

// Only one dynamic bound is constant - fold just that one.
// CHECK-LABEL:   func.func @test_fold_partial_constant_dynamic(
// CHECK-SAME:      %[[X:.*]]: i64, %[[HI:.*]]: i64) -> i64 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min 0
// CHECK-SAME:        max %[[HI]] : i64
// CHECK:           return %[[R]] : i64
// CHECK:         }
func.func @test_fold_partial_constant_dynamic(%x: i64, %hi: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %0 = aster_utils.assume_range %x min %c0 max %hi : i64
  return %0 : i64
}

// i32 type - fold constant dynamic to static.
// CHECK-LABEL:   func.func @test_fold_constant_dynamic_i32(
// CHECK-SAME:      %[[X:.*]]: i32) -> i32 {
// CHECK:           %[[R:.*]] = aster_utils.assume_range %[[X]]
// CHECK-SAME:        min 0
// CHECK-SAME:        max 256 : i32
// CHECK:           return %[[R]] : i32
// CHECK:         }
func.func @test_fold_constant_dynamic_i32(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c256 = arith.constant 256 : i32
  %0 = aster_utils.assume_range %x min %c0 max %c256 : i32
  return %0 : i32
}
