// RUN: aster-opt %s --aster-raise-to-affine | FileCheck %s

// CHECK-DAG: #[[$ATTR_0:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
// CHECK-DAG: #[[$ATTR_1:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-DAG: #[[$ATTR_2:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$ATTR_3:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>

// CHECK-LABEL:   func.func @addi(
// CHECK-SAME:                    %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @addi(%a: index, %b: index, %c: index) -> index {
  %r = aster_utils.addi %a, %b, %c : index
  return %r : index
}

// CHECK-LABEL:   func.func @muli(
// CHECK-SAME:                    %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[ARG0]], %[[ARG1]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @muli(%a: index, %b: index) -> index {
  %r = aster_utils.muli %a, %b : index
  return %r : index
}

// CHECK-LABEL:   func.func @addi_two(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[ARG0]], %[[ARG1]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @addi_two(%a: index, %b: index) -> index {
  %r = aster_utils.addi %a, %b : index
  return %r : index
}

// CHECK-LABEL:   func.func @chain(
// CHECK-SAME:                     %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_3]](){{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]]]
// CHECK:           return %[[APPLY_0]] : index
// CHECK:         }
func.func @chain(%a: index, %b: index, %c: index) -> index {
  %mul = aster_utils.muli %b, %c : index
  %r = aster_utils.addi %a, %mul : index
  return %r : index
}
