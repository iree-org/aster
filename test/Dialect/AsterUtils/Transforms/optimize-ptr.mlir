// RUN: aster-opt %s --aster-optimize-ptr-add --canonicalize --cse | FileCheck %s

// CHECK-LABEL:   func.func @test_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]] const_offset = 42 : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_const_offset(%arg0: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %c42_i64 = arith.constant 42 : i64
  %0 = ptr.ptr_add nuw %arg0, %c42_i64 : !ptr.ptr<#ptr.generic_space>, i64
  return %0 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_uniform_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[ASSUME_RANGE_0]] : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_uniform_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %2 = ptr.ptr_add nuw %arg0, %1 : !ptr.ptr<#ptr.generic_space>, i64
  return %2 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_const_plus_uniform(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[ASSUME_RANGE_0]] const_offset = 16 : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_const_plus_uniform(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %c16_i64 = arith.constant 16 : i64
  %2 = arith.addi %1, %c16_i64 overflow<nsw, nuw> : i64
  %3 = ptr.ptr_add nuw %arg0, %2 : !ptr.ptr<#ptr.generic_space>, i64
  return %3 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_uniform_mul_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[MULI_0]] : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_uniform_mul_const(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %c4_i64 = arith.constant 4 : i64
  %2 = arith.muli %1, %c4_i64 overflow<nsw, nuw> : i64
  %3 = ptr.ptr_add nuw %arg0, %2 : !ptr.ptr<#ptr.generic_space>, i64
  return %3 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_shift_left(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_1]], %[[MULI_0]] : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_shift_left(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %c2_i64 = arith.constant 2 : i64
  %2 = arith.shli %1, %c2_i64 overflow<nsw, nuw> : i64
  %3 = ptr.ptr_add nuw %arg0, %2 : !ptr.ptr<#ptr.generic_space>, i64
  return %3 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_complex_uniform(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[MULI_0]] const_offset = 16 : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_complex_uniform(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %c4_i64 = arith.constant 4 : i64
  %c16_i64 = arith.constant 16 : i64
  %2 = arith.muli %1, %c4_i64 overflow<nsw, nuw> : i64
  %3 = arith.addi %2, %c16_i64 overflow<nsw, nuw> : i64
  %4 = ptr.ptr_add nuw %arg0, %3 : !ptr.ptr<#ptr.generic_space>, i64
  return %4 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_multiple_uniforms(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64,
// CHECK-SAME:      %[[ARG2:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG2]] : i64
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 max 1024 : i64
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[ADDI_0]] : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_multiple_uniforms(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64, %arg2: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %2 = aster_utils.assume_uniform %arg2 : i64
  %3 = aster_utils.assume_range %2 min 0 max 1024 : i64
  %4 = arith.addi %1, %3 overflow<nsw, nuw> : i64
  %5 = ptr.ptr_add nuw %arg0, %4 : !ptr.ptr<#ptr.generic_space>, i64
  return %5 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_no_flags(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ARG1]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_no_flags(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, i64
  return %0 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_nested_add_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]] const_offset = 30 : <#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_nested_add_const(%arg0: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %c10_i64 = arith.constant 10 : i64
  %c20_i64 = arith.constant 20 : i64
  %0 = arith.addi %c10_i64, %c20_i64 overflow<nsw, nuw> : i64
  %1 = ptr.ptr_add nuw %arg0, %0 : !ptr.ptr<#ptr.generic_space>, i64
  return %1 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_i32_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i32
// CHECK:           %[[PTR_ADD_0:.*]] = aster_utils.ptr_add %[[ARG0]], %[[CONSTANT_0]], %[[ASSUME_RANGE_0]] const_offset = 8 : <#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_i32_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i32
  %c8_i32 = arith.constant 8 : i32
  %2 = arith.addi %1, %c8_i32 overflow<nsw, nuw> : i32
  %3 = ptr.ptr_add nuw %arg0, %2 : !ptr.ptr<#ptr.generic_space>, i32
  return %3 : !ptr.ptr<#ptr.generic_space>
}
