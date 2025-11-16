// RUN: aster-opt %s --aster-to-pir | FileCheck %s

// CHECK-LABEL:   func.func @test_add(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_1]], %[[UNREALIZED_CONVERSION_CAST_0]] : <i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>> to i32
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_2]] : i32
// CHECK:         }
func.func @test_add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @test_add_i16(
// CHECK-SAME:      %[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16) -> i16 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i16 to !pir.reg<i16 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i16 to !pir.reg<i16 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i16 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_1]], %[[UNREALIZED_CONVERSION_CAST_0]] : <i16 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i16 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i16 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : !pir.reg<i16 : !amdgcn.vgpr_range<[? + 1]>> to i16
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_2]] : i16
// CHECK:         }
func.func @test_add_i16(%arg0: i16, %arg1: i16) -> i16 {
  %0 = arith.addi %arg0, %arg1 : i16
  return %0 : i16
}

// CHECK-LABEL:   func.func @test_add_i64(
// CHECK-SAME:      %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) -> i64 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i64 to !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i64 to !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_1]], %[[UNREALIZED_CONVERSION_CAST_0]] : <i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>> to i64
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_2]] : i64
// CHECK:         }
func.func @test_add_i64(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL:   func.func @test_add_chained(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> i32 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_2]], %[[UNREALIZED_CONVERSION_CAST_1]] : <i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ALLOCA_1:.*]] = pir.alloca : <i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ADDI_1:.*]] = pir.addi %[[ALLOCA_1]], %[[ADDI_0]], %[[UNREALIZED_CONVERSION_CAST_0]] : <i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[ADDI_1]] : !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>> to i32
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_3]] : i32
// CHECK:         }
func.func @test_add_chained(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.addi %0, %arg2 : i32
  return %1 : i32
}

// CHECK-LABEL:   func.func @test_add_constant(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i32 to !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i32 : !amdgcn.vgpr_range<[? + 1]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_0]], %[[CONSTANT_0]] : <i32 : !amdgcn.vgpr_range<[? + 1]>>, !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>>, i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : !pir.reg<i32 : !amdgcn.vgpr_range<[? + 1]>> to i32
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_1]] : i32
// CHECK:         }
func.func @test_add_constant(%arg0: i32) -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %0 = arith.addi %arg0, %c42_i32 : i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @test_add_index(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : index to !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[ALLOCA_0:.*]] = pir.alloca : <i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[ADDI_0:.*]] = pir.addi %[[ALLOCA_0]], %[[UNREALIZED_CONVERSION_CAST_1]], %[[UNREALIZED_CONVERSION_CAST_0]] : <i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>> to index
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_2]] : index
// CHECK:         }
func.func @test_add_index(%arg0: index, %arg1: index) -> index {
  %0 = arith.addi %arg0, %arg1 : index
  return %0 : index
}
