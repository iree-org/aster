// RUN: aster-opt %s --aster-to-int-arith | FileCheck %s

#map = affine_map<(d0, d1) -> ((d0 + d1) ceildiv 32)>
// CHECK-LABEL:   func.func @test_to_arith(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> (i32, i32) {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 32 : i32
// CHECK-DAG:       %[[THREAD_ID_0:.*]] = aster_utils.thread_id  x
// CHECK-DAG:       %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[THREAD_ID_0]] : i32
// CHECK-DAG:       %[[CMPI_0:.*]] = arith.cmpi sle, %[[ADDI_0]], %[[CONSTANT_1]] : i32
// CHECK-DAG:       %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[ADDI_0]] : i32
// CHECK-DAG:       %[[SUBI_1:.*]] = arith.subi %[[ADDI_0]], %[[CONSTANT_0]] : i32
// CHECK-DAG:       %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[SUBI_0]], %[[SUBI_1]] : i32
// CHECK-DAG:       %[[DIVSI_0:.*]] = arith.divsi %[[SELECT_0]], %[[CONSTANT_2]] : i32
// CHECK-DAG:       %[[SUBI_2:.*]] = arith.subi %[[CONSTANT_1]], %[[DIVSI_0]] : i32
// CHECK-DAG:       %[[ADDI_1:.*]] = arith.addi %[[DIVSI_0]], %[[CONSTANT_0]] : i32
// CHECK-DAG:       %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[SUBI_2]], %[[ADDI_1]] : i32
// CHECK-DAG:       %[[BLOCK_DIM_0:.*]] = aster_utils.block_dim  y
// CHECK-DAG:       return %[[SELECT_1]], %[[BLOCK_DIM_0]] : i32, i32
// CHECK:         }
func.func @test_to_arith(%arg0: index) -> (index, index) {
  %thread_id_x = gpu.thread_id  x
  %0 = affine.apply #map(%arg0, %thread_id_x)
  %block_dim_y = gpu.block_dim  y
  return %0, %block_dim_y : index, index
}
