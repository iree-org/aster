// RUN: aster-opt %s --aster-optimize-arith | FileCheck %s

// CHECK-LABEL:   func.func @test_arith_opt(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32
// CHECK:           %[[THREAD_ID_0:.*]] = aster_utils.thread_id  x
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]]
// CHECK-SAME:        min 1
// CHECK-SAME:        max 32 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[THREAD_ID_0]] overflow<nsw, nuw> : i32
// CHECK:           return %[[ADDI_0]] : i32
// CHECK:         }
func.func @test_arith_opt(%arg0: i32) -> i32 attributes {gpu.block_dims = array<i32: 64, 1, 1>, gpu.grid_dims = array<i32: 1024, 1, 1>, gpu.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c255_i32 = arith.constant 255 : i32
  %0 = aster_utils.thread_id  x
  %1 = aster_utils.assume_range %arg0 min 1 max 32 : i32
  %2 = arith.addi %1, %0 : i32
  %3 = arith.remsi %2, %c255_i32 : i32
  %4 = arith.cmpi slt, %3, %c0_i32 : i32
  %5 = arith.addi %3, %c255_i32 : i32
  %6 = arith.select %4, %5, %3 : i32
  return %6 : i32
}

// Same as above but with constant dynamic bounds - tests that fold/canonicalize
// converts dynamic bounds to static before int-range analysis kicks in.
// CHECK-LABEL:   func.func @test_arith_opt_dynamic_bounds(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32
// CHECK:           %[[THREAD_ID_0:.*]] = aster_utils.thread_id  x
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]]
// CHECK-SAME:        min 1
// CHECK-SAME:        max 32 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[THREAD_ID_0]] overflow<nsw, nuw> : i32
// CHECK:           return %[[ADDI_0]] : i32
// CHECK:         }
func.func @test_arith_opt_dynamic_bounds(%arg0: i32) -> i32 attributes {gpu.block_dims = array<i32: 64, 1, 1>, gpu.grid_dims = array<i32: 1024, 1, 1>, gpu.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c255_i32 = arith.constant 255 : i32
  %0 = aster_utils.thread_id  x
  %1 = aster_utils.assume_range %arg0 min %c1_i32 max %c32_i32 : i32
  %2 = arith.addi %1, %0 : i32
  %3 = arith.remsi %2, %c255_i32 : i32
  %4 = arith.cmpi slt, %3, %c0_i32 : i32
  %5 = arith.addi %3, %c255_i32 : i32
  %6 = arith.select %4, %5, %3 : i32
  return %6 : i32
}

// Dynamic bounds from function args - range is unknown so remsi cannot be
// eliminated. The assume_range should persist with dynamic operands.
// CHECK-LABEL:   func.func @test_arith_opt_truly_dynamic_bounds(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[LO:.*]]: i32, %[[HI:.*]]: i32) -> i32
// CHECK:           %[[ASSUME:.*]] = aster_utils.assume_range %[[ARG0]]
// CHECK-SAME:        min %[[LO]]
// CHECK-SAME:        max %[[HI]] : i32
// CHECK:           arith.remsi
// CHECK:           arith.addi %{{.*}}, %{{.*}} overflow<nsw, nuw> : i32
// CHECK:         }
func.func @test_arith_opt_truly_dynamic_bounds(%arg0: i32, %lo: i32, %hi: i32) -> i32 attributes {gpu.block_dims = array<i32: 64, 1, 1>, gpu.grid_dims = array<i32: 1024, 1, 1>, gpu.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c255_i32 = arith.constant 255 : i32
  %0 = aster_utils.thread_id  x
  %1 = aster_utils.assume_range %arg0 min %lo max %hi : i32
  %2 = arith.addi %1, %0 : i32
  %3 = arith.remsi %2, %c255_i32 : i32
  %4 = arith.cmpi slt, %3, %c0_i32 : i32
  %5 = arith.addi %3, %c255_i32 : i32
  %6 = arith.select %4, %5, %3 : i32
  return %6 : i32
}

// CHECK-LABEL:   func.func @test_nsw_flags(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min -32 max 32 : i8
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG1]] min -32 max 32 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw> : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_nsw_flags(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min -32 max 32 : i8
  %2 = aster_utils.assume_range %arg1 min -32 max 32 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_nuw_flags(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min 0 max -128 : i8
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG1]] min 0 max 55 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nuw> : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_nuw_flags(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min 0 max 128 : i8
  %2 = aster_utils.assume_range %arg1 min 0 max 55 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_invalid_range_1(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 36 : i8
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min -10 max -36 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_invalid_range_1(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min -10 max 220 : i8
  %2 = arith.constant 36 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_invalid_range_2(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 36 : i8
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min -10 max -20 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_invalid_range_2(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min -10 max -20 : i8
  %2 = arith.constant 36 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_invalid_range_3(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 36 : i8
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min 20 max 10 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_invalid_range_3(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min 20 max 10 : i8
  %2 = arith.constant 36 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_invalid_range_4(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 36 : i8
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min -44 max 44 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] overflow<nsw> : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_invalid_range_4(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min -300 max 300 : i8
  %2 = arith.constant 36 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_unsigned_overflow(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 36 : i8
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min 0 max -36 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[CONSTANT_0]] : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_unsigned_overflow(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min 0 max 220 : i8
  %2 = arith.constant 36 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_signed_overflow(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min -32 max 32 : i8
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG1]] min -100 max 32 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_signed_overflow(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min -32 max 32 : i8
  %2 = aster_utils.assume_range %arg1 min -100 max 32 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_nsw_nuw(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i8) -> i8 {
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min 0 max 32 : i8
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG1]] min 32 max 64 : i8
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[ASSUME_RANGE_1]] overflow<nsw, nuw> : i8
// CHECK:           return %[[ADDI_0]] : i8
// CHECK:         }
func.func @test_nsw_nuw(%arg0: i8, %arg1: i8) -> i8 {
  %1 = aster_utils.assume_range %arg0 min 0 max 32 : i8
  %2 = aster_utils.assume_range %arg1 min 32 max 64 : i8
  %3 = arith.addi %1, %2 : i8
  return %3 : i8
}

// CHECK-LABEL:   func.func @test_flags_nsw(
// This test checks a regression where a pattern applied infinitely many times,
// due to the flags not being different from the original flags.
func.func @test_flags_nsw(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %0 = arith.addi %arg0, %c3 overflow<nsw> : i32
  %1 = arith.addi %0, %arg1 overflow<nsw> : i32
  %2 = arith.addi %1, %arg2 overflow<nsw, nuw> : i32
  %3 = arith.addi %2, %c4 overflow<nsw, nuw> : i32
  return %3 : i32
}
