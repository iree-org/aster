// RUN: aster-opt --split-input-file --aster-factorize-affine-expr %s | FileCheck %s

// Simple two-variable factorization: s0*s1 + s2*s1 = (s0+s2)*s1
// CHECK: affine_map<()[s0, s1, s2] -> ((s0 + s2) * s1)>
// CHECK-LABEL: func.func @test_simple
// CHECK: affine.apply #map
func.func @test_simple(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2 * s1)>()[%a, %b, %c]
  return %0 : index
}

// -----

// Full-square factorization: s0*s1 + s2*s1 + s0*s3 + s2*s3 = (s0+s2)*(s1+s3)
// Requires common-factor post-processing.
// CHECK: affine_map<()[s0, s1, s2, s3] -> ((s1 + s3) * (s0 + s2))>
// CHECK-LABEL: func.func @test_square
// CHECK: affine.apply #map
func.func @test_square(%a: index, %b: index, %c: index, %d: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2, s3] ->
    (s0*s1 + s2*s1 + s0*s3 + s2*s3)>()[%a, %b, %c, %d]
  return %0 : index
}

// -----

// Three-variable factorization: s0*s3*s4 + s1*s3*s4 + s2*s3 + s2*s4
//   = ((s0+s1)*s4 + s2)*s3 + s2*s4
// CHECK: affine_map<()[s0, s1, s2, s3, s4] -> (((s0 + s1) * s4 + s2) * s3 + s2 * s4)>
// CHECK-LABEL: func.func @test_three
// CHECK: affine.apply #map
func.func @test_three(%a: index, %b: index, %c: index,
                      %x: index, %y: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2, s3, s4] ->
    (s0*s3*s4 + s1*s3*s4 + s2*s3 + s2*s4)>()[%a, %b, %c, %x, %y]
  return %0 : index
}

// -----

// Already-factored: single product, no change.
// CHECK: affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: func.func @test_no_change
// CHECK: affine.apply #map
func.func @test_no_change(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%a, %b]
  return %0 : index
}

// -----

// Linear expression: no multiplications, no change.
// CHECK: affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func.func @test_linear
// CHECK: affine.apply #map
func.func @test_linear(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%a, %b]
  return %0 : index
}

// -----

// Floordiv sub-expression is recursively factorized.
// (s0*s1 + s2*s1) floordiv 4  →  (s0+s2)*s1 floordiv 4
// CHECK: affine_map<()[s0, s1, s2] -> (((s0 + s2) * s1) floordiv 4)>
// CHECK-LABEL: func.func @test_floordiv_of_factorizable
// CHECK: affine.apply #map
func.func @test_floordiv_of_factorizable(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2] ->
    ((s0 * s1 + s2 * s1) floordiv 4)>()[%a, %b, %c]
  return %0 : index
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0)[s0, s1] -> ((d0 + s1) * s0)>
// CHECK-LABEL:   func.func @test_dims_1(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[ARG0]]){{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @test_dims_1(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s0 * s1)>(%a)[%b, %c]
  return %0 : index
}

// -----

// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0, s1] -> (d0 * (s0 + s1))>
// CHECK-LABEL:   func.func @test_dims_2(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_1]](%[[ARG0]]){{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @test_dims_2(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + d0 * s1)>(%a)[%b, %c]
  return %0 : index
}

// -----

// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0, s1, s2] -> ((s2 + 1) * (s0 + s1))>
// CHECK-LABEL:   func.func @test_constant_term(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @test_constant_term(%a: index, %b: index, %c: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s0 * s2 + s1 + s1 * s2)>()[%a, %b, %c]
  return %0 : index
}
