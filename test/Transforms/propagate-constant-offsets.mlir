// RUN: aster-opt %s --aster-propagate-constant-offsets | FileCheck %s

// Extract constant offset from affine.apply map.
// CHECK-LABEL: func.func @extract_constant(
// CHECK-SAME:    %[[ARG0:.*]]: index) -> index {
// CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : index
// CHECK:         %[[ADD:.*]] = arith.addi %[[ARG0]], %[[C32]] overflow<nsw> : index
// CHECK:         return %[[ADD]] : index
func.func @extract_constant(%arg0: index) -> index {
  %0 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%arg0]
  return %0 : index
}

// Fold arith.addi operands into the map, compose constants.
// (id0 + 1) + (id1 + 2) -> affine.apply(id1 + id0) + 3
// CHECK-LABEL: func.func @fold_and_extract(
// CHECK-SAME:    %[[ID0:.*]]: index, %[[ID1:.*]]: index) -> index {
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[APPLY:.*]] = affine.apply
// CHECK:         %[[ADD:.*]] = arith.addi %[[APPLY]], %[[C3]] overflow<nsw> : index
// CHECK:         return %[[ADD]] : index
func.func @fold_and_extract(%id0: index, %id1: index) -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a0 = arith.addi %id0, %c1 overflow<nsw> : index
  %a1 = arith.addi %id1, %c2 overflow<nsw> : index
  %0 = affine.apply affine_map<(d0)[s0] -> (s0 + d0)>(%a0)[%a1]
  return %0 : index
}

// No constant term - should be unchanged (no arith.addi introduced).
// CHECK-LABEL: func.func @no_constant(
// CHECK-NOT:     arith.addi
// CHECK:         affine.apply
// CHECK:         return
func.func @no_constant(%arg0: index, %arg1: index) -> index {
  %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg0, %arg1)
  return %0 : index
}

// Multiplication with constant offset: s0 * 8 + 42 -> affine(s0 * 8) + 42.
// CHECK-LABEL: func.func @mul_with_offset(
// CHECK-SAME:    %[[ARG:.*]]: index) -> index {
// CHECK-DAG:     %[[C42:.*]] = arith.constant 42 : index
// CHECK:         %[[APPLY:.*]] = affine.apply
// CHECK:         %[[ADD:.*]] = arith.addi %[[APPLY]], %[[C42]] overflow<nsw> : index
// CHECK:         return %[[ADD]] : index
func.func @mul_with_offset(%arg0: index) -> index {
  %0 = affine.apply affine_map<()[s0] -> (s0 * 8 + 42)>()[%arg0]
  return %0 : index
}
