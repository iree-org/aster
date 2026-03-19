// RUN: aster-opt %s --lower-layout-to-affine | FileCheck %s

// CHECK-LABEL: func.func @rank1
// CHECK-SAME:  (%[[C:.*]]: index)
func.func @rank1(%c: index) -> index {
  // Rank-1: delinearize may fold away.
  // CHECK: affine.linearize_index_by_strides[%[[C]]] by (16) : index
  %off = layout.linearize %c, #layout.strided_layout<[64] : [16]>
  return %off : index
}

// CHECK-LABEL: func.func @rank2_rowmajor
// CHECK-SAME:  (%[[C:.*]]: index)
func.func @rank2_rowmajor(%c: index) -> index {
  // Row-major 4x8: stride 8 for rows (slow), stride 1 for cols (fast).
  // CHECK: %[[D:.*]]:2 = affine.delinearize_index %[[C]] into (4, 8)
  // CHECK: affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1] by (8, 1) : index
  %off = layout.linearize %c, #layout.strided_layout<[4, 8] : [8, 1]>
  return %off : index
}

// CHECK-LABEL: func.func @rank2_colmajor
// CHECK-SAME:  (%[[C:.*]]: index)
func.func @rank2_colmajor(%c: index) -> index {
  // Column-major 4x8: stride 1 for rows, stride 4 for cols.
  // CHECK: %[[D:.*]]:2 = affine.delinearize_index %[[C]] into (4, 8)
  // CHECK: affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1] by (1, 4) : index
  %off = layout.linearize %c, #layout.strided_layout<[4, 8] : [1, 4]>
  return %off : index
}

// CHECK-LABEL: func.func @rank3
// CHECK-SAME:  (%[[C:.*]]: index)
func.func @rank3(%c: index) -> index {
  // CHECK: %[[D:.*]]:3 = affine.delinearize_index %[[C]] into (2, 2, 2)
  // CHECK: affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1, %[[D]]#2] by (1, 2, 4) : index
  %off = layout.linearize %c, #layout.strided_layout<[2, 2, 2] : [1, 2, 4]>
  return %off : index
}

// CHECK-LABEL: func.func @nested_rank2
// CHECK-SAME:  (%[[C:.*]]: index)
func.func @nested_rank2(%c: index) -> index {
  // ((2,2),(2,4)):((1,4),(2,8)) flattens to (2,2,2,4):(1,4,2,8)
  // CHECK: %[[D:.*]]:4 = affine.delinearize_index %[[C]] into (2, 2, 2, 4)
  // CHECK: affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1, %[[D]]#2, %[[D]]#3] by (1, 4, 2, 8) : index
  %off = layout.linearize %c, #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>
  return %off : index
}

// CHECK-LABEL: func.func @swizzle
// CHECK-SAME:  (%[[OFF:.*]]: index)
func.func @swizzle(%off: index) -> index {
  // Swizzle(bits=2, base=3, shift=4): mask = ((1<<2)-1) << 3 = 24
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK-DAG: %[[C24:.*]] = arith.constant 24 : i32
  // CHECK: %[[I32:.*]] = arith.index_cast %[[OFF]] : index to i32
  // CHECK: %[[SHR:.*]] = arith.shrui %[[I32]], %[[C4]]
  // CHECK: %[[AND:.*]] = arith.andi %[[SHR]], %[[C24]]
  // CHECK: %[[XOR:.*]] = arith.xori %[[I32]], %[[AND]]
  // CHECK: %[[RES:.*]] = arith.index_cast %[[XOR]] : i32 to index
  // CHECK: return %[[RES]]
  %r = layout.swizzle %off, bits = 2, base = 3, shift = 4
  return %r : index
}
