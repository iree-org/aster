// RUN: aster-opt %s --aster-optimize-arith| FileCheck %s

// CHECK-LABEL:   func.func @test_arith_opt(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 attributes {gpu.block_dims = array<i32: 64, 1, 1>, gpu.grid_dims = array<i32: 1024, 1, 1>, gpu.kernel} {
// CHECK:           %[[THREAD_ID_0:.*]] = aster_utils.thread_id  x
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ARG0]] min 1 max 32 : i32
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
