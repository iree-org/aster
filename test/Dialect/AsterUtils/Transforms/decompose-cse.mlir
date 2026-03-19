// RUN: aster-opt %s --aster-decompose-by-cse --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_three_way_addi(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_3:.*]] = aster_utils.addi %[[ARG4]], %[[PASSTHROUGH_0]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[ADDI_3]] : index, index, index
// CHECK:         }
func.func @test_three_way_addi(%a: index, %b: index, %c: index,
                               %d: index, %e: index) -> (index, index, index) {
  %x = aster_utils.addi %a, %b, %c : index
  %y = aster_utils.addi %a, %b, %d : index
  %z = aster_utils.addi %a, %b, %e : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_no_cse_single_op(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_no_cse_single_op(%a: index, %b: index) -> index {
  %r = aster_utils.addi %a, %b : index
  return %r : index
}

// CHECK-LABEL:   func.func @test_no_cse_disjoint(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[ARG3]] : index
// CHECK:           return %[[ADDI_0]], %[[ADDI_1]] : index, index
// CHECK:         }
func.func @test_no_cse_disjoint(%a: index, %b: index,
                                %c: index, %d: index) -> (index, index) {
  %x = aster_utils.addi %a, %b : index
  %y = aster_utils.addi %c, %d : index
  return %x, %y : index, index
}

// CHECK-LABEL:   func.func @test_identical_ops(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> (index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           return %[[PASSTHROUGH_0]], %[[PASSTHROUGH_0]] : index, index
// CHECK:         }
func.func @test_identical_ops(%a: index, %b: index) -> (index, index) {
  %x = aster_utils.addi %a, %b : index
  %y = aster_utils.addi %a, %b : index
  return %x, %y : index, index
}

// CHECK-LABEL:   func.func @test_two_level(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG2]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[ADDI_1]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG1]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_2]], %[[PASSTHROUGH_1]], %[[PASSTHROUGH_0]] : index, index, index
// CHECK:         }
func.func @test_two_level(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %x = aster_utils.addi %a, %b, %c, %d : index
  %y = aster_utils.addi %a, %c, %d : index
  %z = aster_utils.addi %a, %c : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_three_way_muli(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index) -> (index, index, index) {
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[MULI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_2:.*]] = aster_utils.muli %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_3:.*]] = aster_utils.muli %[[ARG4]], %[[PASSTHROUGH_0]] : index
// CHECK:           return %[[MULI_1]], %[[MULI_2]], %[[MULI_3]] : index, index, index
// CHECK:         }
func.func @test_three_way_muli(%a: index, %b: index, %c: index,
                               %d: index, %e: index) -> (index, index, index) {
  %x = aster_utils.muli %a, %b, %c : index
  %y = aster_utils.muli %a, %b, %d : index
  %z = aster_utils.muli %a, %b, %e : index
  return %x, %y, %z : index, index, index
}

// CHECK-LABEL:   func.func @test_addi_muli_independent(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index) -> (index, index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[MULI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[ARG4]], %[[PASSTHROUGH_1]] : index
// CHECK:           %[[MULI_2:.*]] = aster_utils.muli %[[ARG5]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[MULI_1]], %[[MULI_2]] : index, index, index, index
// CHECK:         }
func.func @test_addi_muli_independent(%a: index, %b: index, %c: index,
                                      %d: index, %e: index, %f: index) -> (index, index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d : index
  %xm = aster_utils.muli %a, %b, %e : index
  %ym = aster_utils.muli %a, %b, %f : index
  return %xa, %ya, %xm, %ym : index, index, index, index
}

// CHECK-LABEL:   func.func @test_muli_not_cse_single(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_2]], %[[MULI_0]] : index, index, index
// CHECK:         }
func.func @test_muli_not_cse_single(%a: index, %b: index, %c: index,
                                    %d: index) -> (index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d : index
  %xm = aster_utils.muli %a, %b : index
  return %xa, %ya, %xm : index, index, index
}

// CHECK-LABEL:   func.func @test_chained_cse(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index) {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[ARG2]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[ADDI_1]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[ADDI_2]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_3:.*]] = aster_utils.addi %[[ARG3]], %[[PASSTHROUGH_1]] : index
// CHECK:           %[[ADDI_4:.*]] = aster_utils.addi %[[ADDI_3]], %[[PASSTHROUGH_1]] : index
// CHECK:           return %[[ADDI_1]], %[[ADDI_3]], %[[ADDI_4]] : index, index, index
// CHECK:         }
func.func @test_chained_cse(%a: index, %b: index, %c: index, %d: index) -> (index, index, index) {
  %xa = aster_utils.addi %a, %b, %c : index
  %ya = aster_utils.addi %a, %b, %d, %xa : index
  %xm = aster_utils.addi %a, %b, %xa, %ya : index
  return %xa, %ya, %xm : index, index, index
}

// -----

// This test covers a regression where the pass produced an op with a single operand, causing a verification error.
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<()[s0] -> (s0 floordiv 16)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
#map = affine_map<()[s0] -> ((s0 mod 64) floordiv 4)>
#map1 = affine_map<()[s0] -> (s0 floordiv 4)>
#map2 = affine_map<()[s0] -> (s0 floordiv 8)>
#map3 = affine_map<()[s0] -> (s0 floordiv 16)>
#map4 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
// CHECK-LABEL:   func.func @test_single_expr_regression() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -512 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 32 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant -32768 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4096 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant -32 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 256 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 8 : index
// CHECK:           %[[BLOCK_ID_0:.*]] = gpu.block_id x
// CHECK:           %[[THREAD_ID_0:.*]] = gpu.thread_id x
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[THREAD_ID_0]], %[[CONSTANT_6]] : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[THREAD_ID_0]]]
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[APPLY_0]], %[[CONSTANT_5]] : index
// CHECK:           %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[THREAD_ID_0]]]
// CHECK:           %[[MULI_2:.*]] = aster_utils.muli %[[APPLY_1]], %[[CONSTANT_4]] : index
// CHECK:           %[[APPLY_2:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[BLOCK_ID_0]]]
// CHECK:           %[[MULI_3:.*]] = aster_utils.muli %[[APPLY_2]], %[[CONSTANT_3]] : index
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[MULI_0]], %[[MULI_2]] : index
// CHECK:           %[[PASSTHROUGH_0:.*]] = aster_utils.passthrough %[[ADDI_0]] tag = "__decompose_ops__" : index
// CHECK:           %[[ADDI_1:.*]] = aster_utils.addi %[[MULI_3]], %[[MULI_1]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_1:.*]] = aster_utils.passthrough %[[ADDI_1]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_4:.*]] = aster_utils.muli %[[BLOCK_ID_0]], %[[CONSTANT_3]] : index
// CHECK:           %[[MULI_5:.*]] = aster_utils.muli %[[APPLY_2]], %[[CONSTANT_2]] : index
// CHECK:           %[[ADDI_2:.*]] = aster_utils.addi %[[MULI_4]], %[[MULI_5]], %[[MULI_1]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[PASSTHROUGH_2:.*]] = aster_utils.passthrough %[[ADDI_2]] tag = "__decompose_ops__" : index
// CHECK:           %[[MULI_6:.*]] = aster_utils.muli %[[APPLY_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[ADDI_3:.*]] = aster_utils.addi %[[MULI_6]], %[[PASSTHROUGH_0]] : index
// CHECK:           %[[MULI_7:.*]] = aster_utils.muli %[[THREAD_ID_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[APPLY_3:.*]] = affine.apply #[[$ATTR_3]](){{\[}}%[[THREAD_ID_0]]]
// CHECK:           %[[MULI_8:.*]] = aster_utils.muli %[[APPLY_3]], %[[CONSTANT_0]] : index
// CHECK:           %[[APPLY_4:.*]] = affine.apply #[[$ATTR_4]](){{\[}}%[[THREAD_ID_0]]]
// CHECK:           %[[MULI_9:.*]] = aster_utils.muli %[[APPLY_4]], %[[CONSTANT_6]] : index
// CHECK:           %[[ADDI_4:.*]] = aster_utils.addi %[[MULI_7]], %[[MULI_8]], %[[MULI_9]] : index
// CHECK:           %[[ADDI_5:.*]] = aster_utils.addi %[[CONSTANT_1]], %[[PASSTHROUGH_1]] : index
// CHECK:           %[[ADDI_6:.*]] = aster_utils.addi %[[CONSTANT_1]], %[[PASSTHROUGH_2]] : index
// CHECK:           return
// CHECK:         }
func.func @test_single_expr_regression() {
  %c-512 = arith.constant -512 : index
  %c32 = arith.constant 32 : index
  %c-32768 = arith.constant -32768 : index
  %c4096 = arith.constant 4096 : index
  %c-32 = arith.constant -32 : index
  %c256 = arith.constant 256 : index
  %c8 = arith.constant 8 : index
  %block_id_x = gpu.block_id x
  %thread_id_x = gpu.thread_id x
  %0 = aster_utils.muli %thread_id_x, %c8 : index
  %1 = affine.apply #map()[%thread_id_x]
  %2 = aster_utils.muli %1, %c256 : index
  %3 = affine.apply #map1()[%thread_id_x]
  %4 = aster_utils.muli %3, %c-32 : index
  %5 = affine.apply #map2()[%block_id_x]
  %6 = aster_utils.muli %5, %c4096 : index
  %7 = aster_utils.addi %0, %2, %4, %6 : index
  %8 = aster_utils.muli %block_id_x, %c4096 : index
  %9 = aster_utils.muli %5, %c-32768 : index
  %10 = aster_utils.addi %0, %8, %2, %4, %9 : index
  %11 = aster_utils.muli %1, %c32 : index
  %12 = aster_utils.addi %0, %11, %4 : index
  %13 = aster_utils.muli %thread_id_x, %c32 : index
  %14 = affine.apply #map3()[%thread_id_x]
  %15 = aster_utils.muli %14, %c-512 : index
  %16 = affine.apply #map4()[%thread_id_x]
  %17 = aster_utils.muli %16, %c8 : index
  %18 = aster_utils.addi %13, %15, %17 : index
  %19 = aster_utils.addi %0, %2, %4, %6, %c32 : index
  %20 = aster_utils.addi %0, %8, %2, %4, %9, %c32 : index
  return
}
