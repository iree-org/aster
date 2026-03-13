// RUN: aster-opt %s --aster-expand-affine-apply --cse | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-LABEL:   func.func @test_add_two(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_add_two(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%a, %b]
  return %0 : index
}

// CHECK-LABEL:   func.func @test_add_four(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_add_four(%a: index, %b: index, %c: index, %d: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%a, %b, %c, %d]
  return %0 : index
}


// CHECK-LABEL:   func.func @test_mul_const(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[CONSTANT_0]] : index
// CHECK:           return %[[MULI_0]] : index
// CHECK:         }
func.func @test_mul_const(%a: index) -> index {
  %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%a]
  return %0 : index
}


// CHECK-LABEL:   func.func @test_mixed_add_mul(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[MULI_0:.*]] = aster_utils.muli %[[ARG0]], %[[CONSTANT_0]] : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK:           %[[MULI_1:.*]] = aster_utils.muli %[[ARG1]], %[[CONSTANT_1]] : index
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[MULI_0]], %[[MULI_1]], %[[ARG2]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_mixed_add_mul(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 2 + s2)>()[%a, %b, %c]
  return %0 : index
}


// CHECK-LABEL:   func.func @test_constant() -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : index
// CHECK:           return %[[CONSTANT_0]] : index
// CHECK:         }
func.func @test_constant() -> index {
  %0 = affine.apply affine_map<() -> (42)>()
  return %0 : index
}


// CHECK-LABEL:   func.func @test_dim_and_sym(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           return %[[ADDI_0]] : index
// CHECK:         }
func.func @test_dim_and_sym(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%a)[%b]
  return %0 : index
}


// CHECK-LABEL:   func.func @test_floordiv_of_add(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[ADDI_0:.*]] = aster_utils.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ADDI_0]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @test_floordiv_of_add(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> ((s0 + s1) floordiv 4)>()[%a, %b]
  return %0 : index
}
