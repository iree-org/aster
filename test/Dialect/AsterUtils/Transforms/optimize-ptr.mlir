// RUN: aster-opt %s --aster-optimize-ptr-add --canonicalize --cse | FileCheck %s

// CHECK-LABEL:   func.func @test_const_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_const_offset(%arg0: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %c42_i64 = arith.constant 42 : i64
  %0 = ptr.ptr_add nuw %arg0, %c42_i64 : !ptr.ptr<#ptr.generic_space>, i64
  return %0 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_uniform_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[ASSUME_RANGE_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_uniform_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i64
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i64
  %2 = ptr.ptr_add nuw %arg0, %1 : !ptr.ptr<#ptr.generic_space>, i64
  return %2 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_const_plus_uniform(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[ASSUME_RANGE_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add nuw %[[PTR_ADD_0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i64
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i64
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i64
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add nuw %[[PTR_ADD_0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i64
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i64
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG2]] : i64
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 max 1024 : i64
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[ADDI_0]] : !ptr.ptr<#ptr.generic_space>, i64
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ARG1]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_no_flags(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) -> !ptr.ptr<#ptr.generic_space> {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr.ptr<#ptr.generic_space>, i64
  return %0 : !ptr.ptr<#ptr.generic_space>
}
// CHECK-LABEL:   func.func @test_nested_add_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 30 : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i64
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
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 max 1024 : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add nuw %[[ARG0]], %[[ASSUME_RANGE_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add nuw %[[PTR_ADD_0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_i32_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 max 1024 : i32
  %c8_i32 = arith.constant 8 : i32
  %2 = arith.addi %1, %c8_i32 overflow<nsw, nuw> : i32
  %3 = ptr.ptr_add nuw %arg0, %2 : !ptr.ptr<#ptr.generic_space>, i32
  return %3 : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL:   func.func @test_additive_separable_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : i32
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG2]] min 0 : i32
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG3]] : i32
// CHECK:           %[[ASSUME_RANGE_2:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[ASSUME_RANGE_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[PTR_ADD_1]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_additive_separable_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32, %arg2: i32, %arg3: i32) -> !ptr.ptr<#ptr.generic_space> {
  %c64_i32 = arith.constant 64 : i32
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 : i32
  %2 = aster_utils.assume_range %arg2 min 0 : i32
  %3 = aster_utils.assume_uniform %arg3 : i32
  %4 = aster_utils.assume_range %3 min 0 : i32
  %5 = arith.addi %1, %2 overflow<nsw> : i32
  %6 = arith.addi %5, %4 overflow<nsw> : i32
  %7 = arith.addi %6, %c64_i32 overflow<nsw> : i32
  %8 = ptr.ptr_add %arg0, %7 : !ptr.ptr<#ptr.generic_space>, i32
  return %8 : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL:   func.func @test_multiplicative_separable_offset_1(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : i32
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG2]] min 0 : i32
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG3]] : i32
// CHECK:           %[[ASSUME_RANGE_2:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_1]], %[[ASSUME_RANGE_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_multiplicative_separable_offset_1(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32, %arg2: i32, %arg3: i32) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 : i32
  %2 = aster_utils.assume_range %arg2 min 0 : i32
  %3 = aster_utils.assume_uniform %arg3 : i32
  %4 = aster_utils.assume_range %3 min 0 : i32
  %5 = arith.addi %1, %2 overflow<nsw> : i32
  %6 = arith.muli %5, %4 overflow<nsw> : i32
  %7 = ptr.ptr_add %arg0, %6 : !ptr.ptr<#ptr.generic_space>, i32
  return %7 : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL:   func.func @test_multiplicative_separable_offset_2(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 16 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : i32
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG2]] min 0 : i32
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG3]] : i32
// CHECK:           %[[ASSUME_RANGE_2:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_1]], %[[ASSUME_RANGE_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[MULI_0]], %[[CONSTANT_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_3:.*]] = arith.muli %[[MULI_2]], %[[CONSTANT_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_4:.*]] = arith.muli %[[ASSUME_RANGE_2]], %[[CONSTANT_0]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_3]], %[[MULI_4]] overflow<nsw, nuw> : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[PTR_ADD_1]], %[[CONSTANT_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_multiplicative_separable_offset_2(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32, %arg2: i32, %arg3: i32) -> !ptr.ptr<#ptr.generic_space> {
  %c16_i32 = arith.constant 16 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 : i32
  %2 = aster_utils.assume_range %arg2 min 0 : i32
  %3 = aster_utils.assume_uniform %arg3 : i32
  %4 = aster_utils.assume_range %3 min 0 : i32
  %5 = arith.addi %1, %2 overflow<nsw> : i32
  %6 = arith.addi %5, %c32_i32 overflow<nsw> : i32
  %7 = arith.muli %6, %4 overflow<nsw> : i32
  %8 = arith.muli %7, %c2_i32 overflow<nsw> : i32
  %9 = arith.addi %8, %c16_i32 overflow<nsw> : i32
  %10 = ptr.ptr_add %arg0, %9 : !ptr.ptr<#ptr.generic_space>, i32
  return %10 : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL:   func.func @test_multiplicative_separable_offset_3(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 256 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 32 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 16 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : i32
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG2]] min 0 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_1]], %[[ADDI_1]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[ASSUME_RANGE_1]], %[[ADDI_2]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[MULI_0]], %[[MULI_1]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[CONSTANT_1]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_3:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_0]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_4:.*]] = arith.addi %[[MULI_2]], %[[MULI_3]] overflow<nsw, nuw> : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_4]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[ADDI_3]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[PTR_ADD_1]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_multiplicative_separable_offset_3(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32, %arg2: i32, %arg3: i32) -> !ptr.ptr<#ptr.generic_space> {
  %c16_i32 = arith.constant 16 : i32
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 : i32
  %2 = aster_utils.assume_range %arg2 min 0 : i32
  %3 = arith.addi %1, %2 overflow<nsw> : i32
  %4 = arith.addi %3, %c16_i32 overflow<nsw> : i32
  %5 = arith.muli %4, %4 overflow<nsw> : i32
  %6 = ptr.ptr_add %arg0, %5 : !ptr.ptr<#ptr.generic_space>, i32
  return %6 : !ptr.ptr<#ptr.generic_space>
}

// CHECK-LABEL:   func.func @test_multiplicative_non_separable_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : i32
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : i32
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG2]] min 0 : i32
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG3]] : i32
// CHECK:           %[[ASSUME_RANGE_2:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_1]] min 0 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw, nuw> : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_2]], %[[CONSTANT_0]] overflow<nsw, nuw> : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[MULI_0]], %[[ADDI_0]] overflow<nsw, nuw> : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_multiplicative_non_separable_offset(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i32, %arg2: i32, %arg3: i32) -> !ptr.ptr<#ptr.generic_space> {
  %c16_i32 = arith.constant 16 : i32
  %0 = aster_utils.assume_uniform %arg1 : i32
  %1 = aster_utils.assume_range %0 min 0 : i32
  %2 = aster_utils.assume_range %arg2 min 0 : i32
  %3 = aster_utils.assume_uniform %arg3 : i32
  %4 = aster_utils.assume_range %3 min 0 : i32
  %5 = arith.muli %1, %2 overflow<nsw> : i32
  %6 = arith.addi %4, %c16_i32 overflow<nsw> : i32
  %7 = arith.muli %5, %6 overflow<nsw> : i32
  %8 = ptr.ptr_add %arg0, %7 : !ptr.ptr<#ptr.generic_space>, i32
  return %8 : !ptr.ptr<#ptr.generic_space>
}
