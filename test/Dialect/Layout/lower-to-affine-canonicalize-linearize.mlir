// RUN: aster-opt %s --lower-layout-to-affine --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @bounded_rank2_no_use
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index) -> index
// CHECK:         %[[R:.*]] = affine.linearize_index disjoint [%[[A]], %[[B]]] by (4) : index
// CHECK:         return %[[R]] : index
func.func @bounded_rank2_no_use(%a: index, %b: index) -> index {
  %r = affine.linearize_index disjoint [%a, %b] by (64, 4) : index
  return %r : index
}

// CHECK-LABEL: func.func @strides_rank2_no_use
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index) -> index
// CHECK:         %[[R:.*]] = affine.linearize_index disjoint [%[[A]], %[[B]]] by (4) : index
// CHECK:         return %[[R]] : index
func.func @strides_rank2_no_use(%a: index, %b: index) -> index {
  %r = affine.linearize_index_by_strides[%a, %b] by (4, 1) : index
  return %r : index
}

// CHECK-LABEL: func.func @bounded_rank3_with_apply
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[BASE:.*]]: index) -> index
// CHECK:         %[[L:.*]] = affine.linearize_index disjoint [%[[A]], %[[B]], %[[C]]] by (8, 4) : index
// CHECK:         %[[R:.*]] = affine.apply {{.*}}[%[[BASE]], %[[L]]]
// CHECK:         return %[[R]] : index
func.func @bounded_rank3_with_apply(%a: index, %b: index, %c: index, %base: index) -> index {
  %l = affine.linearize_index disjoint [%a, %b, %c] by (16, 8, 4) : index
  %r = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%l)[%base]
  return %r : index
}

// CHECK-LABEL: func.func @strides_rank3_with_apply
// CHECK-SAME:    (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[BASE:.*]]: index) -> index
// CHECK:         %[[L:.*]] = affine.linearize_index disjoint [%[[A]], %[[B]], %[[C]]] by (8, 4) : index
// CHECK:         %[[R:.*]] = affine.apply {{.*}}[%[[BASE]], %[[L]]]
// CHECK:         return %[[R]] : index
func.func @strides_rank3_with_apply(%a: index, %b: index, %c: index, %base: index) -> index {
  // suffix_product((8, 4)) = (32, 4, 1)
  %l = affine.linearize_index_by_strides[%a, %b, %c] by (32, 4, 1) : index
  %r = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0)>(%l)[%base]
  return %r : index
}

// Roundtrip: delinearize_index then linearize_index folds to the original
// value for both forms, and the fold is preserved across the normalization.

// CHECK-LABEL: func.func @bounded_roundtrip
// CHECK-SAME:    (%[[X:.*]]: index) -> index
// CHECK-NEXT:    return %[[X]] : index
func.func @bounded_roundtrip(%x: index) -> index {
  %d:2 = affine.delinearize_index %x into (4, 8) : index, index
  %r = affine.linearize_index disjoint [%d#0, %d#1] by (4, 8) : index
  return %r : index
}

// CHECK-LABEL: func.func @strides_roundtrip
// CHECK-SAME:    (%[[X:.*]]: index) -> index
// CHECK-NEXT:    return %[[X]] : index
func.func @strides_roundtrip(%x: index) -> index {
  %d:2 = affine.delinearize_index %x into (4, 8) : index, index
  %r = affine.linearize_index_by_strides[%d#0, %d#1] by (8, 1) : index
  return %r : index
}
