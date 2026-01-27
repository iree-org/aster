// RUN: aster-opt %s --verify-roundtrip
// RUN: aster-opt %s | FileCheck %s

// -----
// Test struct type parsing and printing

// CHECK-LABEL: func.func @test_struct_type_simple
// CHECK-SAME: %arg0: !aster_utils.struct<x: i32, y: f32>
func.func @test_struct_type_simple(%arg0: !aster_utils.struct<x: i32, y: f32>) -> !aster_utils.struct<x: i32, y: f32> {
  return %arg0 : !aster_utils.struct<x: i32, y: f32>
}

// CHECK-LABEL: func.func @test_struct_type_tensor
// CHECK-SAME: %arg0: !aster_utils.struct<a: tensor<4xf16>, b: tensor<4xf16>>
func.func @test_struct_type_tensor(%arg0: !aster_utils.struct<a: tensor<4xf16>, b: tensor<4xf16>>) -> !aster_utils.struct<a: tensor<4xf16>, b: tensor<4xf16>> {
  return %arg0 : !aster_utils.struct<a: tensor<4xf16>, b: tensor<4xf16>>
}

// CHECK-LABEL: func.func @test_struct_type_single_field
// CHECK-SAME: %arg0: !aster_utils.struct<value: i64>
func.func @test_struct_type_single_field(%arg0: !aster_utils.struct<value: i64>) -> !aster_utils.struct<value: i64> {
  return %arg0 : !aster_utils.struct<value: i64>
}

// CHECK-LABEL: func.func @test_struct_type_empty
// CHECK-SAME: %arg0: !aster_utils.struct<>
func.func @test_struct_type_empty(%arg0: !aster_utils.struct<>) -> !aster_utils.struct<> {
  return %arg0 : !aster_utils.struct<>
}

// -----
// Test struct_create operation

// CHECK-LABEL: func.func @test_struct_create
// CHECK: %[[S:.*]] = aster_utils.struct_create(%arg0, %arg1) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
func.func @test_struct_create(%x: i32, %y: f32) -> !aster_utils.struct<x: i32, y: f32> {
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  return %s : !aster_utils.struct<x: i32, y: f32>
}

// CHECK-LABEL: func.func @test_struct_create_single
// CHECK: %[[S:.*]] = aster_utils.struct_create(%arg0) : (i64) -> !aster_utils.struct<value: i64>
func.func @test_struct_create_single(%v: i64) -> !aster_utils.struct<value: i64> {
  %s = aster_utils.struct_create(%v) : (i64) -> !aster_utils.struct<value: i64>
  return %s : !aster_utils.struct<value: i64>
}

// CHECK-LABEL: func.func @test_struct_create_empty
// CHECK: %[[S:.*]] = aster_utils.struct_create() : () -> !aster_utils.struct<>
func.func @test_struct_create_empty() -> !aster_utils.struct<> {
  %s = aster_utils.struct_create() : () -> !aster_utils.struct<>
  return %s : !aster_utils.struct<>
}

// -----
// Test struct_extract operation

// CHECK-LABEL: func.func @test_struct_extract_x
// CHECK: aster_utils.struct_extract %arg0 ["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
func.func @test_struct_extract_x(%s: !aster_utils.struct<x: i32, y: f32>) -> i32 {
  %x = aster_utils.struct_extract %s ["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
  return %x : i32
}

// CHECK-LABEL: func.func @test_struct_extract_y
// CHECK: aster_utils.struct_extract %arg0 ["y"] : !aster_utils.struct<x: i32, y: f32> -> f32
func.func @test_struct_extract_y(%s: !aster_utils.struct<x: i32, y: f32>) -> f32 {
  %y = aster_utils.struct_extract %s ["y"] : !aster_utils.struct<x: i32, y: f32> -> f32
  return %y : f32
}

// CHECK-LABEL: func.func @test_struct_extract_multiple
// CHECK: aster_utils.struct_extract %arg0 ["x", "y"] : !aster_utils.struct<x: i32, y: f32> -> i32, f32
func.func @test_struct_extract_multiple(%s: !aster_utils.struct<x: i32, y: f32>) -> (i32, f32) {
  %x, %y = aster_utils.struct_extract %s ["x", "y"] : !aster_utils.struct<x: i32, y: f32> -> i32, f32
  return %x, %y : i32, f32
}

// -----
// Test combined create and extract

// CHECK-LABEL: func.func @test_struct_roundtrip
func.func @test_struct_roundtrip(%x: i32, %y: f32) -> (i32, f32) {
  // CHECK: aster_utils.struct_create
  %s = aster_utils.struct_create(%x, %y) : (i32, f32) -> !aster_utils.struct<x: i32, y: f32>
  // CHECK: aster_utils.struct_extract {{.*}} ["x"]
  %x2 = aster_utils.struct_extract %s ["x"] : !aster_utils.struct<x: i32, y: f32> -> i32
  // CHECK: aster_utils.struct_extract {{.*}} ["y"]
  %y2 = aster_utils.struct_extract %s ["y"] : !aster_utils.struct<x: i32, y: f32> -> f32
  return %x2, %y2 : i32, f32
}

// -----
// Test nested struct types (struct containing tensors)

// CHECK-LABEL: func.func @test_nested_tensor_struct
func.func @test_nested_tensor_struct(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: aster_utils.struct_create
  %s = aster_utils.struct_create(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> !aster_utils.struct<lhs: tensor<4x4xf32>, rhs: tensor<4x4xf32>>
  // CHECK: aster_utils.struct_extract {{.*}} ["lhs"]
  %lhs = aster_utils.struct_extract %s ["lhs"] : !aster_utils.struct<lhs: tensor<4x4xf32>, rhs: tensor<4x4xf32>> -> tensor<4x4xf32>
  return %lhs : tensor<4x4xf32>
}
