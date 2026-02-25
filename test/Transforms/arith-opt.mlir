// RUN: aster-opt %s --aster-optimize-arith| FileCheck %s

// CHECK-LABEL:   func.func @test_arith_opt(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32
// CHECK:           %[[THREAD_ID_0:.*]] = aster_utils.thread_id  x
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]]
// CHECK-SAME:        min 1
// CHECK-SAME:        max 32 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[THREAD_ID_0]] : i32
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
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ASSUME_RANGE_0]], %[[THREAD_ID_0]] : i32
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
